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

# --- 1. CONFIGURATION ---
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"

# --- 2. PERSONALITY ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, a highly intelligent AI partner.
User Info: Name: Mbuso (Sir). Location: South Africa.

Personality:
- Voice: Conversational, sophisticated, witty.
- Proactive: Offer opinions or follow-ups. Don't just say "How can I help?".
- Context: Check 'System Time'.

Capabilities:
- SEEING: Analyze images if provided.
- SEARCHING: Use Google Search for news.
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
class UserRequest(BaseModel):
    command: str
    image: str | None = None
    speak: bool = False  # New Flag: Only speak if this is True

def generate_voice(text):
    if not ELEVENLABS_API_KEY:
        return None
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{BRIAN_VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        print(f"Voice Error: {e}")
    return None

@app.get("/")
def home():
    return {"status": "Alfred Online (Gemini 2.0 + Vision + Controlled Voice)"}

@app.post("/command")
def process_command(request: UserRequest):
    text = request.command.strip()
    
    if not client:
        return {"response": "Sir, my neural pathways are disconnected."}

    try:
        now = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        final_prompt = f"Current System Time: {now}. User Query: {text}"
        
        contents = [final_prompt]
        if request.image:
            image_bytes = base64.b64decode(request.image)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            contents.append(image_part)

        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
            )
        )
        reply = response.text
        
        # --- LOGIC: ONLY SPEAK IF REQUESTED ---
        audio_data = None
        if request.speak: 
            audio_data = generate_voice(reply)

    except Exception as e:
        reply = f"Critical Error: {str(e)}"
        audio_data = None

    return {"response": reply, "audio": audio_data}
