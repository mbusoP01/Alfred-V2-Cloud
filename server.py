# server.py
import os
import base64
import json
import requests
import re
import io
import time
import pytz
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- LIBRARIES ---
from fpdf import FPDF
from docx import Document
from pptx import Presentation
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from duckduckgo_search import DDGS

# --- KEYS ---
SERVER_SECRET_KEY = "Mbuso.08@"
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

MEMORY_FILE = "alfred_memory.json"

# --- MEMORY ENGINE ---
def load_memory():
    """Reads the long-term memory file."""
    if not os.path.exists(MEMORY_FILE):
        return "No prior knowledge of the user."
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
            # Convert list of facts to a string
            return "\n".join([f"- {fact}" for fact in data.get("facts", [])])
    except Exception as e:
        return f"Memory Error: {str(e)}"

def save_memory_fact(fact):
    """Writes a new fact to the memory file."""
    try:
        data = {"facts": []}
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
        
        # Avoid duplicates
        if fact not in data["facts"]:
            data["facts"].append(fact)
            
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Memory Saved: {fact}")
    except Exception as e:
        print(f"Memory Save Error: {e}")

# --- SYSTEM PROMPT (DYNAMIC) ---
def get_system_instructions():
    user_memory = load_memory()
    return f"""
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

*** LONG-TERM MEMORY (DO NOT FORGET) ***
{user_memory}

PROTOCOL:
1. USE REAL-WORLD DATA: If 'SEARCH_DATA' is provided, use it.
2. MEMORY UPDATES: If the user tells you a NEW permanent fact about themselves (name, preference, job, car, etc.), you MUST save it.
   - Output format: <<<MEM_SAVE>>> The user's car is a BMW 320d <<<MEM_END>>>
   - Do not save trivial things like "I am hungry". Save permanent things like "I am allergic to peanuts".

*** CRITICAL: FILE GENERATION ***
If user asks to create/generate a file:
1. Do NOT chat.
2. Wrap content in: <<<FILE_START>>> content <<<FILE_END>>>.

*** CRITICAL: IMAGE GENERATION ***
If asked to generate an image:
1. Reply exactly: "IMAGE_GEN_REQUEST: [Detailed Prompt]"
"""

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

class ChatMessage(BaseModel):
    role: str
    content: str

class UserRequest(BaseModel):
    command: str
    file_data: Optional[str] = None
    mime_type: Optional[str] = None
    speak: bool = False
    thinking_mode: bool = False
    history: List[ChatMessage] = []

# --- 1. RESEARCHER ENGINE ---
def perform_web_search(query):
    print(f"Searching Web for: {query}")
    try:
        results = DDGS().text(query, max_results=4)
        if not results: return None
        summary = "REAL-TIME SEARCH DATA FOUND:\n"
        for r in results:
            summary += f"- Title: {r['title']}\n  Link: {r['href']}\n  Snippet: {r['body']}\n\n"
        return summary
    except: return None

# --- 2. LINK READER ---
def get_url_content(text):
    url_match = re.search(r'(https?://[^\s]+)', text)
    if not url_match: return None
    url = url_match.group(0)
    try:
        if "youtube.com" in url or "youtu.be" in url:
            video_id = url.split("v=")[1].split("&")[0] if "v=" in url else url.split("/")[-1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([t['text'] for t in transcript])
            return f"[YOUTUBE TRANSCRIPT]: {full_text[:10000]}"
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]): script.extract()
        return f"[WEBSITE CONTENT]: {soup.get_text(separator=' ', strip=True)[:10000]}"
    except: return None

# --- 3. IMAGE GEN ---
def generate_high_quality_image(prompt):
    if not HUGGINGFACE_API_KEY: return None, None
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    models = ["https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"]
    for model_url in models:
        try:
            response = requests.post(model_url, headers=headers, json={"inputs": prompt}, timeout=15)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8'), f"gen_{int(time.time())}.jpg"
        except: continue
    return None, None

# --- 4. FILE FACTORY ---
def create_file(content, file_type):
    try:
        buffer = io.BytesIO()
        filename = f"alfred_doc_{int(time.time())}"
        if file_type == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1'))
            return base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('utf-8'), f"{filename}.pdf"
        elif file_type == "docx":
            doc = Document()
            doc.add_heading('Alfred Generation', 0)
            doc.add_paragraph(content)
            doc.save(buffer)
            return base64.b64encode(buffer.getvalue()).decode('utf-8'), f"{filename}.docx"
        elif file_type == "pptx":
            prs = Presentation()
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = "Alfred Presentation"
            slide.placeholders[1].text = content
            prs.save(buffer)
            return base64.b64encode(buffer.getvalue()).decode('utf-8'), f"{filename}.pptx"
        elif file_type == "txt":
            return base64.b64encode(content.encode('utf-8')).decode('utf-8'), f"{filename}.txt"
    except: return None, None

# --- 5. CHART ENGINE ---
def execute_chart_code(code):
    try:
        local_env = {"plt": plt, "np": np}
        exec(code, {}, local_env)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8'), "alfred_chart.png"
    except: return None, None

# --- 6. VOICE ENGINE ---
def generate_voice(text):
    if not ELEVENLABS_API_KEY: return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{BRIAN_VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    # Remove tags before speaking
    clean = re.sub(r'<<<.*?>>>', '', text).replace('IMAGE_GEN_REQUEST:', '')
    if not clean.strip(): return None
    try:
        res = requests.post(url, json={"text": clean[:800], "model_id": "eleven_monolingual_v1"}, headers=headers)
        if res.status_code == 200: return base64.b64encode(res.content).decode('utf-8')
    except: pass
    return None

@app.post("/command")
def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY: raise HTTPException(status_code=401)
    if not client: return {"response": "System Failure: No Gemini API Key."}

    try:
        sa_time = datetime.now(pytz.timezone('Africa/Johannesburg')).strftime("%Y-%m-%d %H:%M")
        
        # Identify File Request
        req_type = next((t for t in ["pdf", "docx", "pptx", "txt"] if t in request.command.lower()), None)
        if "word" in request.command.lower(): req_type = "docx"
        
        # Prepare History
        history = [types.Content(role="model" if m.role=="alfred" else "user", parts=[types.Part.from_text(text=m.content)]) for m in request.history]

        # Context Building
        context = ""
        scraped = get_url_content(request.command)
        if scraped: context += f"\n{scraped}\n"
        
        triggers = ["search", "find", "news", "latest", "price", "who is", "what is"]
        if not scraped and any(x in request.command.lower() for x in triggers):
            res = perform_web_search(request.command)
            if res: context += f"\n{res}\n"

        # Final Prompt
        prompt = f"[Time: {sa_time}] {context} \nUser: {request.command}"
        parts = [prompt]
        if request.file_data:
            parts.append(types.Part.from_bytes(data=base64.b64decode(request.file_data), mime_type=request.mime_type))

        # Generate
        chat = client.chats.create(
            model='gemini-2.0-flash-thinking-exp-01-21' if request.thinking_mode else 'gemini-2.0-flash',
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=get_system_instructions(), # Loads memory dynamically
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        
        response = chat.send_message(parts).text
        
        # --- OUTPUT PROCESSING ---
        gen_file, gen_name = None, None
        
        # 1. Memory Save
        if "<<<MEM_SAVE>>>" in response:
            try:
                fact = response.split("<<<MEM_SAVE>>>")[1].split("<<<MEM_END>>>")[0].strip()
                save_memory_fact(fact)
                # Remove tag from user view
                response = re.sub(r'<<<MEM_SAVE>>>.*?<<<MEM_END>>>', '', response, flags=re.DOTALL).strip()
                if not response: response = "Memory updated, Sir."
            except: pass

        # 2. File Gen
        if "<<<FILE_START>>>" in response:
            try:
                content = response.split("<<<FILE_START>>>")[1].split("<<<FILE_END>>>")[0].strip()
                response = "Document manufactured, Sir."
                gen_file, gen_name = create_file(content, req_type or "txt")
            except: response = "File assembly failed."

        # 3. Image Gen
        elif "IMAGE_GEN_REQUEST:" in response:
            prompt = response.split("IMAGE_GEN_REQUEST:")[1].strip()
            response = "Rendering visual..."
            gen_file, gen_name = generate_high_quality_image(prompt)

        # 4. Chart Gen
        elif "<<<CHART_START>>>" in response:
            try:
                code = response.split("<<<CHART_START>>>")[1].split("<<<CHART_END>>>")[0].strip()
                response = "Visualizing data..."
                gen_file, gen_name = execute_chart_code(code)
            except: pass

        audio = generate_voice(response) if request.speak else None

    except Exception as e:
        print(f"Error: {e}")
        response = f"System Error: {str(e)}"
        audio, gen_file, gen_name = None, None, None

    return {"response": response, "audio": audio, "file": {"data": gen_file, "name": gen_name} if gen_file else None}
