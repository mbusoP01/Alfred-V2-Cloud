# server.py
import os
import base64
import requests
import re
from google import genai
from google.genai import types
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- CONFIGURATION ---
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"

ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

PROTOCOL:
1. Answer the CURRENT query directly.
2. Use conversation history for context.
3. Use Google Search for current events.
4. IF A FILE IS UPLOADED: Analyze it thoroughly. Summarize or answer questions based on it.

IMAGE GENERATION:
If asked to generate/create an image:
1. Output ONLY a Markdown Image Link.
2. Format: ![Description](https://image.pollinations.ai/prompt/{description_with_underscores}?nologo=true)
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
    file_data: Optional[str] = None # Renamed from 'image' to 'file_data'
    mime_type: Optional[str] = None # New field to know if it's PDF or Image
    speak: bool = False
    history: List[ChatMessage] = []

# --- VOICE HELPER ---
def generate_voice(text):
    if not ELEVENLABS_API_KEY: return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{BRIAN_VOICE_ID}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    
    clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    clean_text = clean_text.replace("*", "").replace("#", "").replace("`", "")
    if not clean_text.strip(): return None

    data = {"text": clean_text[:1000], "model_id": "eleven_monolingual_v1", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    try:
        res = requests.post(url, json=data, headers=headers)
        if res.status_code == 200: return base64.b64encode(res.content).decode('utf-8')
    except Exception: pass
    return None

@app.get("/")
def home():
    return {"status": "Alfred Online (Gemini 2.0 + Files + Voice)"}

@app.post("/command")
def process_command(request: UserRequest):
    if not client: return {"response": "Sir, connection severed (No API Key)."}

    try:
        now = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        
        # History
        chat_history = []
        for msg in request.history:
            role = "model" if msg.role == "alfred" else "user"
            chat_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        # Current Message + File (Image OR PDF)
        current_parts = [f"[System Time: {now}] {request.command}"]
        
        if request.file_data and request.mime_type:
            try:
                file_bytes = base64.b64decode(request.file_data)
                # Pass the specific mime type (e.g., 'application/pdf' or 'image/jpeg')
                file_part = types.Part.from_bytes(data=file_bytes, mime_type=request.mime_type)
                current_parts.append(file_part)
            except Exception:
                pass

        # Generate
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
        audio_data = generate_voice(reply) if request.speak else None

    except Exception as e:
        reply = f"Processing Error: {str(e)}"
        audio_data = None

    return {"response": reply, "audio": audio_data}
