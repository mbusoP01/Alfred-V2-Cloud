# server.py
import os
import google.generativeai as genai
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. PERSONAL KNOWLEDGE BASE (EDIT THIS!) ---
# This is where you teach Alfred about Mbuso.
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an advanced AI assistant.
User Info:
- Name: Mbuso (Sir).
- Location: South Africa.
- Interests: Coding, Upcycling, Business.
- Personality: Helpful, formal, and concise.

Capabilities:
- You have access to Google Search. If asked about current events (like G20, News, Weather), USE IT.
- Always check the provided 'System Time' to understand when 'now' is.
"""
# ----------------------------------------------

api_key = os.environ.get("GEMINI_API_KEY")
current_model_name = "Unknown"

if api_key:
    genai.configure(api_key=api_key)
    try:
        # Auto-detect the best model
        all_models = [m.name for m in genai.list_models()]
        supported_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        best_model = next((m for m in supported_models if 'flash' in m), None)
        if not best_model and supported_models:
            best_model = supported_models[0]
            
        current_model_name = best_model if best_model else "No Models Found"
        
        if best_model:
            # --- 2. CONNECT TO INTERNET (The Search Tool) ---
            # We enable 'google_search_retrieval' so he can look up 2025 news
            tools_list = 'google_search_retrieval'
            
            model = genai.GenerativeModel(
                model_name=best_model,
                tools=tools_list,
                system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
            )
        else:
            model = None

    except Exception as e:
        current_model_name = "Error: " + str(e)
        model = None
else:
    model = None
    current_model_name = "No API Key"

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
    return {
        "status": "Alfred Online", 
        "model": current_model_name,
        "tools": "Google Search Enabled"
    }

@app.post("/command")
def process_command(request: UserRequest):
    text = request.command.strip()
    
    if not model:
        return {"response": f"Sir, I am offline. Diagnosis: {current_model_name}."}

    try:
        # Time Injection
        now = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        final_prompt = f"Current System Time: {now}. User Query: {text}"
        
        # Ask the AI (It will decide if it needs to Search Google)
        response = model.generate_content(final_prompt)
        reply = response.text
    except Exception as e:
        reply = f"I tried to access the web but failed. Error: {str(e)}"

    return {"response": reply}
