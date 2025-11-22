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
    # The new SDK client
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
    status = "Online (New SDK)" if client else "Offline (No Key)"
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
        
        # --- THE NEW SEARCH TOOL CONFIGURATION ---
        # This is the exact syntax for the new google-genai SDK
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp', # Trying the latest, falling back to 1.5 if needed
            contents=final_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
            )
        )
        
        reply = response.text
    except Exception as e:
        # Fallback to stable 1.5 if 2.0 fails
        try:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
                )
            )
            reply = response.text
        except Exception as e2:
            reply = f"I tried to search but encountered a critical error: {str(e2)}"

    return {"response": reply}
