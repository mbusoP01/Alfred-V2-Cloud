# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import random

app = FastAPI()

# --- CONFIGURATION: ALLOW CONNECTION FROM ANYWHERE (CORS) ---
# This is the "Security Gate" that was missing!
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,  # <--- I set this to False for you to fix the security bug
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------------------------------

class UserRequest(BaseModel):
    command: str

@app.get("/")
def home():
    # If you see this message, you know you have the NEW code
    return {"status": "Alfred v2 Brain is operational."}

@app.post("/command")
def process_command(request: UserRequest):
    text = request.command.lower().strip()
    response_text = ""

    # --- ALFRED LOGIC BLOCK ---
    
    # 1. GREETINGS
    if any(word in text for word in ["hello", "hi", "hey"]):
        greetings = ["At your service, sir.", "Systems online.", "Hello, sir."]
        response_text = random.choice(greetings)

    # 2. TIME & DATE
    elif "time" in text:
        now = datetime.now().strftime("%H:%M")
        response_text = f"The current time is {now}."
    elif "date" in text:
        today = datetime.now().strftime("%A, %B %d, %Y")
        response_text = f"Today is {today}."

    # 3. CALCULATOR
    elif "calculate" in text or "calc" in text:
        try:
            expression = text.replace("calculate", "").replace("calc", "")
            result = eval(expression) 
            response_text = f"The result is {result}."
        except:
            response_text = "I could not calculate that, sir."

    # 4. FALLBACK
    else:
        response_text = "I do not recognize that command yet, sir."

    return {"response": response_text}
