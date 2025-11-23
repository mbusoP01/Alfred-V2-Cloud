import os
import base64
import requests
import re
import io
import time
import logging
import asyncio
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- STABLE LIBRARY ---
import google.generativeai as genai

# --- KEYS ---
SERVER_SECRET_KEY = "Mbuso.08@"
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    logger.warning("GEMINI_API_KEY is missing! The server will not function correctly.")
else:
    genai.configure(api_key=api_key)

# --- SYSTEM INSTRUCTIONS ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

*** PRIMARY PROTOCOL ***
1. Answer the CURRENT query directly and concisely.
2. **REAL-TIME DATA:** You have the Google Search tool enabled. Use it to find real-time info for:
   - Stocks (e.g., "Price of AAPL")
   - News
   - Weather
   - Sport scores

*** IMAGE PROTOCOL ***
1. If the user asks for an image generation: Reply "IMAGE_GEN_REQUEST: [Detailed Prompt]"
2. If the user wants to see an existing image: Use Google Search to find a URL and Reply: <<<FETCH_IMAGE>>>[Link]<<<FETCH_IMAGE>>>

*** FILE PROTOCOL ***
To generate a file, use this format exactly: <<<FILE_START>>> content <<<FILE_END>>>.
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class UserRequest(BaseModel):
    command: str
    file_data: Optional[str] = None
    mime_type: Optional[str] = None
    history: List[ChatMessage] = []

# --- HELPER FUNCTIONS ---

def fetch_and_encode_image(url):
    try:
        from PIL import Image
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), "fetched.jpg", "image/jpeg"
    except Exception as e:
        logger.error(f"Image fetch error: {e}")
        return None, None, None

def create_file(content, file_type):
    buffer = io.BytesIO()
    timestamp = datetime.now().strftime('%H%M%S')
    filename = f"alfred_doc_{timestamp}"
    try:
        if file_type == "pdf":
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            clean_content = content.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 7, clean_content)
            return base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('utf-8'), f"{filename}.pdf"
        elif file_type == "docx":
            from docx import Document
            doc = Document()
            doc.add_paragraph(content)
            doc.save(buffer)
            return base64.b64encode(buffer.getvalue()).decode('utf-8'), f"{filename}.docx"
        else: # TXT
            return base64.b64encode(content.encode('utf-8')).decode('utf-8'), f"{filename}.txt"
    except Exception as e:
        logger.error(f"File creation error: {e}")
        return None, None
    return None, None

# --- MAIN ENDPOINT ---
@app.post("/command")
async def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # 1. Prepare History (Stable SDK format)
        history_gemini = []
        for m in request.history:
            role = "user" if m.role == "user" else "model"
            history_gemini.append({"role": role, "parts": [m.content]})

        # 2. Configure Model (Using Gemini 1.5 Flash which is very stable with Tools)
        # We enable the built-in Google Search tool
        tools = [
            {"google_search": {}} 
        ]
        
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            tools=tools,
            system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
        )
        
        # 3. Chat Session
        chat = model.start_chat(history=history_gemini)
        
        # 4. Generate Response (Retry Loop)
        response_text = "I am having trouble connecting."
        for attempt in range(3):
            try:
                response = chat.send_message(request.command)
                response_text = response.text
                break
            except Exception as e:
                if "429" in str(e): # Rate Limit
                    time.sleep(2)
                    continue
                logger.error(f"Gemini API Error: {e}")
                response_text = f"Error: {str(e)}"
                break

        # 5. Handle Actions (Image/File)
        gen_file_data, gen_filename, gen_mime = None, None, None

        img_match = re.search(r'<<<FETCH_IMAGE>>>(.*?)<<<FETCH_IMAGE>>>', response_text, re.DOTALL)
        if img_match:
            url = img_match.group(1).strip()
            response_text = response_text.replace(img_match.group(0), " [Image Fetched] ").strip()
            b64, fname, mime = fetch_and_encode_image(url)
            if b64: 
                gen_file_data, gen_filename, gen_mime = b64, fname, mime

        elif "<<<FILE_START>>>" in response_text:
            try:
                parts = response_text.split("<<<FILE_START>>>")
                pre_text = parts[0]
                content = parts[1].split("<<<FILE_END>>>")[0].strip()
                ftype = "txt"
                if "pdf" in request.command.lower(): ftype = "pdf"
                elif "word" in request.command.lower(): ftype = "docx"
                gen_file_data, gen_filename = create_file(content, ftype)
                if gen_file_data:
                    gen_mime = "application/pdf" if ftype == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    response_text = pre_text + f"\n[Attached: {gen_filename}]"
            except: pass

        return {
            "response": response_text,
            "file": {"data": gen_file_data, "name": gen_filename, "mime": gen_mime} if gen_file_data else None
        }

    except Exception as e:
        return {"response": f"Critical Server Error: {str(e)}"}
