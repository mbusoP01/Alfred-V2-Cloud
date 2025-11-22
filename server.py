# server.py
import os
import google.generativeai as genai
from google.generativeai import types # <--- CRITICAL IMPORT
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
- You have access to Google Search. Use it for current events, news, or facts you don't know.
- If the user asks a personal question (like "how are you"), you do NOT need to search. Just answer politely.
- Always check the provided 'System Time' to understand when 'now' is.
"""

# --- 2. SETUP & AUTO-DISCOVERY ---
api_key = os.environ.get("GEMINI_API_KEY")
current_model_name = "Unknown"

if api_key:
    genai.configure(api_key=api_key)
    try:
        # Find models that support content generation
        all_models = [m.name for m in genai.list_models()]
        supported_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        # Prefer the newest 'flash' model
        best_model = next((m for m in supported_models if 'flash' in m), None)
        if not best_model and supported_models:
            best_model = supported_models[0]
            
        current_model_name = best_model if best_model else "No Models Found"
        
        if best_model:
            # --- THE FIX IS HERE ---
            # We use the official 'types' wrapper to avoid the FunctionDeclaration error
            google_search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            model = genai.GenerativeModel(
                model_name=best_model,
                tools=[google_search_tool], # <--- Passed as a proper Tool object
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
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        
        # Ask the AI
        response = model.generate_content(final_prompt)
        reply = response.text
    except Exception as e:
        reply = f"I encountered a processing error, Sir. Details: {str(e)}"

    return {"response": reply}
