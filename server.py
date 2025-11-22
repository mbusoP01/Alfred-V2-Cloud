# server.py
import os
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- SETUP AI BRAIN ---
# This pulls the key you just saved in Render
api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    # We use 'gemini-1.5-flash' because it is fast and smart
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None
# ----------------------

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
    return {"status": "Alfred v2 (AI Powered) is online."}

@app.post("/command")
def process_command(request: UserRequest):
    text = request.command.strip()
    
    # Safety Check
    if not model:
        return {"response": "Sir, my API key is missing. Please check Render settings."}

    try:
        # Ask the AI
        ai_response = model.generate_content(text)
        reply = ai_response.text
    except Exception as e:
        reply = f"I encountered an error, Sir: {str(e)}"

    return {"response": reply}
