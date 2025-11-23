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
# To enable Voice: Add 'ELEVENLABS_API_KEY' to Render Environment Variables
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"  # The "Brian" Voice

# --- 2. NEW PERSONALITY INSTRUCTIONS ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, a highly intelligent and proactive AI partner.
User Info:
- Name: Mbuso (Sir).
- Location: South Africa.

Personality & Tone:
- Voice: You are NOT a robotic search engine. You are a conversationalist.
- Proactive: Do not always end with "How can I help?". Instead, offer an opinion, a relevant fact, or a follow-up idea.
- Style: Sophisticated, slightly witty, and warm. Like a trusted advisor, not a customer service bot.
- Context: Always check 'System Time' to understand current events.

Capabilities:
- SEEING: You can see images. Describe them naturally if asked.
- SEARCHING: Use Google Search for news/events.
- SPEAKING: Your text is read aloud. Keep sentences rhythmically pleasing.
"""

# --- 3. SETUP CLIENTS ---
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

class UserRequest(BaseModel):
    command: str
    image: str | None = None

# --- HELPER: TEXT-TO-SPEECH ---
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
            # Return audio as base64 string
            return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        print(f"Voice Error: {e}")
    return None

@app.get("/")
def home():
    status = "Online (Gemini 2.0)"
    if ELEVENLABS_API_KEY:
        status += " + Voice Module (Brian)"
    return {"status": status}

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

        # ASK GEMINI
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
            )
        )
        reply = response.text
        
        # GENERATE VOICE
        audio_data = generate_voice(reply)

    except Exception as e:
        reply = f"Critical Error: {str(e)}"
        audio_data = None

    return {"response": reply, "audio": audio_data}
