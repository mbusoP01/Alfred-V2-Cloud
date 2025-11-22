# server.py
import os
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CONFIGURATION ---
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

app = FastAPI()

# --- SECURITY GATE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------

class UserRequest(BaseModel):
    command: str

@app.get("/")
def home():
    # DIAGNOSTIC: This will list available models on the home page
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"status": "Alfred Online", "available_models": available}
    except:
        return {"status": "Alfred Online", "available_models": "Could not fetch list"}

@app.post("/command")
def process_command(request: UserRequest):
    text = request.command.strip()
    
    if not api_key:
        return {"response": "Error: API Key is missing in Render settings."}

    # --- SMART MODEL SWITCHER ---
    # We try the fast model first. If it fails (404), we switch to the stable backup.
    try:
        # Attempt 1: Try specific Flash version
        model = genai.GenerativeModel('gemini-1.5-flash-001')
        response = model.generate_content(text)
        reply = response.text
        
    except Exception as e:
        try:
            print(f"Flash failed: {e}. Switching to backup...")
            # Attempt 2: Backup Model (Gemini Pro 1.0 - Very Stable)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(text)
            reply = response.text
        except Exception as e2:
            # If both fail, show the error
            reply = f"I am having trouble accessing my brain circuits. Error: {str(e2)}"

    return {"response": reply}
