# server.py
import os
import base64
import requests
import re
import io
import time
from google import genai
from google.genai import types
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- FILE LIBRARIES ---
from fpdf import FPDF
from docx import Document
from pptx import Presentation

# --- SECURITY & KEYS ---
SERVER_SECRET_KEY = "Mbuso.08@"
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

# --- MODEL CONFIGURATION ---
# We use FLUX.1-dev (The current King of Open Source)
# Fallback: stable-diffusion-xl-base-1.0 if Flux is busy
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

PROTOCOL:
1. Answer the CURRENT query directly.
2. IF A FILE IS UPLOADED: Analyze it thoroughly.
3. IMAGE GEN: If asked to generate/create an image:
   - Reply exactly: "IMAGE_GEN_REQUEST: [Detailed Prompt based on user request]"
   - Do not generate a link. The server will handle the heavy rendering.
"""

api_key = os.environ.get("GEMINI_API_KEY")
client = None
if api_key:
    client = genai.Client(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
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
    speak: bool = False
    history: List[ChatMessage] = []

# --- IMAGE GENERATION ENGINE (FLUX) ---
def generate_high_quality_image(prompt):
    if not HUGGINGFACE_API_KEY:
        return None, None
    
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(HF_IMAGE_URL, headers=headers, json=payload)
        
        # If model is loading, wait a bit and try again (Common with Free Tier)
        if "error" in response.json() and "loading" in response.json()["error"]:
            time.sleep(5) # Wait for model to load
            response = requests.post(HF_IMAGE_URL, headers=headers, json=payload)

        if response.status_code == 200:
            # Return the raw image bytes
            image_b64 = base64.b64encode(response.content).decode('utf-8')
            filename = f"flux_gen_{datetime.now().strftime('%H%M%S')}.jpg"
            return image_b64, filename
    except Exception as e:
        print(f"Image Gen Error: {e}")
        pass
    return None, None

# --- FILE FACTORY ---
def create_file(content, file_type):
    buffer = io.BytesIO()
    filename = f"alfred_doc_{datetime.now().strftime('%H%M%S')}"
    if file_type == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        safe_text = content.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_text)
        return base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('utf-8'), f"{filename}.pdf"
    elif file_type == "docx":
        doc = Document()
        doc.add_paragraph(content)
        doc.save(buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), f"{filename}.docx"
    elif file_type == "pptx":
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.placeholders[1].text = content
        prs.save(buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), f"{filename}.pptx"
    elif file_type == "txt":
        return base64.b64encode(content.encode('utf-8')).decode('utf-8'), f"{filename}.txt"
    return None, None

# --- VOICE HELPER ---
def generate_voice(text):
    if not ELEVENLABS_API_KEY: return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{BRIAN_VOICE_ID}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    clean_text = re.sub(r'IMAGE_GEN_REQUEST:.*', 'Generating image now, Sir.', text)
    clean_text = clean_text.replace("*", "").replace("#", "").replace("`", "")
    if not clean_text.strip(): return None
    try:
        res = requests.post(url, json={"text": clean_text[:1000], "model_id": "eleven_monolingual_v1"}, headers=headers)
        if res.status_code == 200: return base64.b64encode(res.content).decode('utf-8')
    except: pass
    return None

@app.get("/")
def home():
    return {"status": "Alfred Online (Flux Vision)"}

@app.post("/command")
def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY:
        raise HTTPException(status_code=401, detail="ACCESS DENIED")

    if not client: return {"response": "Sir, connection severed (No API Key)."}

    try:
        now = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        requested_type = None
        lower_cmd = request.command.lower()
        if "pdf" in lower_cmd: requested_type = "pdf"
        elif "word" in lower_cmd or "docx" in lower_cmd: requested_type = "docx"
        elif "presentation" in lower_cmd or "pptx" in lower_cmd: requested_type = "pptx"
        elif "text file" in lower_cmd or "txt" in lower_cmd: requested_type = "txt"

        chat_history = []
        for msg in request.history:
            role = "model" if msg.role == "alfred" else "user"
            chat_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        current_parts = [f"[System Time: {now}] {request.command}"]
        if request.file_data:
            try:
                file_bytes = base64.b64decode(request.file_data)
                current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=request.mime_type))
            except: pass

        chat_session = client.chats.create(
            model='gemini-2.0-flash',
            history=chat_history,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
            )
        )
        
        response = chat_session.send_message(message=current_parts)
        reply = response.text
        
        gen_file_data = None
        gen_filename = None

        # --- HANDLE FILE GEN ---
        if "FILE_CONTENT_START" in reply and requested_type:
            content = reply.split("FILE_CONTENT_START")[1].split("FILE_CONTENT_END")[0].strip()
            reply = "I have generated the document for you, Sir."
            gen_file_data, gen_filename = create_file(content, requested_type)

        # --- HANDLE IMAGE GEN (FLUX) ---
        elif "IMAGE_GEN_REQUEST:" in reply:
            image_prompt = reply.split("IMAGE_GEN_REQUEST:")[1].strip()
            reply = "I am rendering the image using the Flux engine, Sir. This may take a moment."
            gen_file_data, gen_filename = generate_high_quality_image(image_prompt)
            if not gen_file_data:
                reply = "I attempted to generate the image, but the Flux engine is currently overloaded. Please try again in 30 seconds."

        audio_data = generate_voice(reply) if request.speak else None

    except Exception as e:
        reply = f"Processing Error: {str(e)}"
        audio_data, gen_file_data, gen_filename = None, None, None

    return {"response": reply, "audio": audio_data, "file": {"data": gen_file_data, "name": gen_filename} if gen_file_data else None}
