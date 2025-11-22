# server.py
import os
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
- You have access to Google Search. Use it for current events, news, or facts you don't know.
- If asked about current events (like "Who won the rugby world cup 2023"), SEARCH.
- Always check the provided 'System Time'.
"""

# --- 2. SETUP NEW CLIENT ---
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
# ---------------------

class UserRequest(BaseModel):
    command: str

@app.get("/")
def home():
    status = "Online (Gemini 2.0)" if client else "Offline (No Key)"
    return {"status": status, "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/command")
def process_command(request: UserRequest):
    text = request.command.strip()
    
    if not client:
        return {"response": "Sir, my neural pathways are disconnected (API Key missing)."}

    try:
        # Time Injection
        now = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        final_prompt = f"Current System Time: {now}. User Query: {text}"
        
        # --- THE FIX: GEMINI 2.0 ONLY ---
        # We removed the fallback to 1.5 because it causes the 404 error.
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=final_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
            )
        )
        
        reply = response.text
    except Exception as e:
        reply = f"I encountered a critical error, Sir: {str(e)}"

    return {"response": reply}
