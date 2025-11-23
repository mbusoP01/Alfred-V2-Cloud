# server.py
import os
import base64
import requests
import re
import io
from google import genai
from google.genai import types
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- FILE LIBRARIES ---
from fpdf import FPDF
from docx import Document
from pptx import Presentation

# --- CONFIGURATION ---
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"

ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

PROTOCOL:
1. Answer the CURRENT query directly.
2. If the user asks to GENERATE A FILE (PDF, DOCX, PPTX, TXT):
   - Do NOT just write the text.
   - Instead, write the CONTENT that should go into the file.
   - Start your response with "FILE_CONTENT_START" and end with "FILE_CONTENT_END".
   - Example: User asks "Make a pdf about cars". You reply: "Certainly. FILE_CONTENT_START Cars are fast... FILE_CONTENT_END"
   - The server will handle the file creation.
"""

# --- SETUP CLIENT ---
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

# --- FILE FACTORY ---
def create_file(content, file_type):
    buffer = io.BytesIO()
    filename = f"alfred_doc_{datetime.now().strftime('%H%M%S')}"
    
    if file_type == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # sanitize unicode for simple PDF gen
        safe_text = content.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_text)
        pdf_out = pdf.output(dest='S').encode('latin-1')
        return base64.b64encode(pdf_out).decode('utf-8'), f"{filename}.pdf"

    elif file_type == "docx":
        doc = Document()
        doc.add_heading('Alfred Generation', 0)
        doc.add_paragraph(content)
        doc.save(buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), f"{filename}.docx"

    elif file_type == "pptx":
        prs = Presentation()
        slide_layout = prs.slide_layouts[1] # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content_box = slide.placeholders[1]
        title.text = "Alfred Presentation"
        content_box.text = content
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
    clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', text).replace("*", "")
    if not clean_text.strip(): return None
    data = {"text": clean_text[:1000], "model_id": "eleven_monolingual_v1", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    try:
        res = requests.post(url, json=data, headers=headers)
        if res.status_code == 200: return base64.b64encode(res.content).decode('utf-8')
    except: pass
    return None

@app.get("/")
def home():
    return {"status": "Alfred Online (Files + Voice + Vision)"}

@app.post("/command")
def process_command(request: UserRequest):
    if not client: return {"response": "Sir, connection severed (No API Key)."}

    try:
        now = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        
        # Check if user asked for a file
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
                file_part = types.Part.from_bytes(data=file_bytes, mime_type=request.mime_type)
                current_parts.append(file_part)
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
        
        # --- FILE EXTRACTION ---
        generated_file_data = None
        generated_filename = None
        
        if "FILE_CONTENT_START" in reply and requested_type:
            # Extract content between markers
            content = reply.split("FILE_CONTENT_START")[1].split("FILE_CONTENT_END")[0].strip()
            # Clean reply for chat display
            reply = reply.replace("FILE_CONTENT_START", "").replace("FILE_CONTENT_END", "").replace(content, "I have generated the file for you, Sir.")
            # Generate File
            generated_file_data, generated_filename = create_file(content, requested_type)

        audio_data = generate_voice(reply) if request.speak else None

    except Exception as e:
        reply = f"Processing Error: {str(e)}"
        audio_data = None
        generated_file_data = None
        generated_filename = None

    return {
        "response": reply, 
        "audio": audio_data,
        "file": {"data": generated_file_data, "name": generated_filename} if generated_file_data else None
    }
