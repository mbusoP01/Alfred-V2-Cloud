# server.py
import os
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- SETUP & AUTO-DISCOVERY ---
api_key = os.environ.get("GEMINI_API_KEY")
current_model_name = "Unknown"

if api_key:
    genai.configure(api_key=api_key)
    try:
        # 1. Ask Google: "What models do you have?"
        all_models = [m.name for m in genai.list_models()]
        
        # 2. Filter: We only want models that can generate text
        supported_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # 3. Pick the Best: Look for 'flash' (fastest), otherwise take the first one
        # This logic finds the newest 'flash' model automatically
        best_model = next((m for m in supported_models if 'flash' in m), None)
        
        if not best_model and supported_models:
            best_model = supported_models[0] # Fallback to anything available
            
        current_model_name = best_model if best_model else "No Models Found"
        
        # 4. Configure the AI with the winner
        if best_model:
            model = genai.GenerativeModel(best_model)
        else:
            model = None

    except Exception as e:
        print(f"Error finding models: {e}")
        model = None
        current_model_name = "Error: " + str(e)
else:
    model = None
    current_model_name = "No API Key"
# ------------------------------

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
    # This will tell you EXACTLY which model Alfred decided to use
    return {
        "status": "Alfred Online", 
        "brain_model": current_model_name
    }

@app.post("/command")
def process_command(request: UserRequest):
    text = request.command.strip()
    
    if not model:
        return {"response": f"Sir, I cannot think. Diagnosis: {current_model_name}. Check Render Logs."}

    try:
        # Ask the Auto-Selected Model
        response = model.generate_content(text)
        reply = response.text
    except Exception as e:
        reply = f"My brain ({current_model_name}) malfunctioned. Error: {str(e)}"

    return {"response": reply}
