# server.py
import os
import base64
import requests
from google import genai
from google.genai import types
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- 1. CONFIGURATION ---
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"

# --- 2. PERSONALITY & INSTRUCTIONS ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

PROTOCOL:
1. FOCUS: Answer the CURRENT query directly. Do NOT summarize past messages unless asked.
2. BREVITY: Be concise. Avoid fluff. Get straight to the point.
3. FORMAT: Use Markdown. Use code blocks for code.
4. MEMORY: Use conversation history only for context (e.g., if user says "change it", know what "it" is).
5. TOOLS: Use Google Search for current events (e.g., 2024/2025 news).
"""

# --- 3. SETUP CLIENT ---
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

# --- UPDATED DATA MODEL ---
class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    content: str

class UserRequest(BaseModel):
    command: str
    image: Optional[str] = None
    speak: bool = False
    history: List[ChatMessage] = [] # Alfred now accepts history!

# --- HELPER: TEXT-TO-SPEECH ---
def generate_voice(text):
    if not ELEVENLABS_API_KEY: return None
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{BRIAN_VOICE_ID}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    # We strip markdown symbols for cleaner speech
    clean_text = text.replace("*", "").replace("#", "").replace("`", "")
    data = {
        "text": clean_text[:1000], # Limit voice char count to save credits
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    try:
        res = requests.post(url, json=data, headers=headers)
        if res.status_code == 200:
            return base64.b64encode(res.content).decode('utf-8')
    except Exception:
        pass
    return None

@app.get("/")
def home():
    return {"status": "Alfred Online (Memory + Voice + Vision)"}

@app.post("/command")
def process_command(request: UserRequest):
    if not client: return {"response": "Sir, connection severed (No API Key)."}

    try:
        now = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        
        # --- 1. PREPARE MESSAGE & HISTORY ---
        # We build the chat session using the history sent from the phone
        chat_history = []
        for msg in request.history:
            # Map frontend 'alfred'/'user' to Gemini 'model'/'user'
            role = "model" if msg.role == "alfred" else "user"
            chat_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        # Current Message
        current_parts = [f"[System Time: {now}] {request.command}"]
        if request.image:
            img_bytes = base64.b64decode(request.image)
            current_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        # --- 2. GENERATE ---
        # We use generate_content but provide history as context if needed, 
        # or use chats.create() logic. For simplicity/reliability in REST, 
        # we append history to the contents or use the chat interface.
        # The 2.0 SDK handles lists of contents well.
        
        # Let's use the chat interface for memory awareness
        chat = client.chats.create(
            model='gemini-2.0-flash',
            history=chat_history,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
            )
        )
        
        response = chat.send_message(message=current_parts)
        reply = response.text
        
        # --- 3. VOICE ---
        audio_data = generate_voice(reply) if request.speak else None

    except Exception as e:
        reply = f"Processing Error: {str(e)}"
        audio_data = None

    return {"response": reply, "audio": audio_data}
