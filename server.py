# server.py
import os
import base64
from google import genai
from google.genai import types
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. SYSTEM INSTRUCTIONS ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, a sophisticated AI assistant.
User Info:
- Name: Mbuso (Sir).
- Location: South Africa.
- Personality: Helpful, formal, and concise.

Capabilities:
- You can SEE images. If the user sends an image, analyze it.
- You have access to Google Search. Use it for current events.
- Always check the provided 'System Time'.
"""

# --- 2. SETUP CLIENT ---
api_key = os.environ.get("GEMINI_API_KEY")
client = None

if api_key:
    client = genai.Client(api_key=api_key)

app = FastAPI()

# --- SECURITY GATE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. DATA MODELS ---
class UserRequest(BaseModel):
    command: str
    image: str | None = None  # New field for the image code

@app.get("/")
def home():
    status = "Online (Gemini 2.0 + Vision)" if client else "Offline (No Key)"
    return {"status": status}

@app.post("/command")
def process_command(request: UserRequest):
    text = request.command.strip()
    
    if not client:
        return {"response": "Sir, my neural pathways are disconnected."}

    try:
        now = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        prompt_text = f"Current System Time: {now}. User Query: {text}"
        
        # --- PREPARE CONTENTS ---
        contents = [prompt_text]

        # If there is an image, add it to the package
        if request.image:
            try:
                image_bytes = base64.b64decode(request.image)
                # Create a special Image Part for Gemini
                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg" 
                )
                contents.append(image_part)
            except Exception as e:
                return {"response": f"Error processing image visual: {str(e)}"}

        # --- SEND TO GEMINI 2.0 ---
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
            )
        )
        
        reply = response.text
    except Exception as e:
        reply = f"I encountered a critical error, Sir: {str(e)}"

    return {"response": reply}
