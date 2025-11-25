# server.py
import os
import base64
import json
import requests
import re
import io
import time
import pytz
import asyncio
import edge_tts
import traceback
import random
from github import Github
from google import genai
from google.genai import types
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union

# --- LIBRARIES ---
from fpdf import FPDF
from docx import Document
from pptx import Presentation
from pptx.util import Pt as PptxPt, Inches
from pptx.dml.color import RGBColor as PptxColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from duckduckgo_search import DDGS

# --- KEYS ---
SERVER_SECRET_KEY = "Mbuso.08@"
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO_NAME = os.environ.get("GITHUB_REPO")
MEMORY_FILE = "alfred_memory.json"

# --- CLOUD MEMORY ---
def pull_memory_from_cloud():
    if not GITHUB_TOKEN or not GITHUB_REPO_NAME: return
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO_NAME)
        contents = repo.get_contents(MEMORY_FILE)
        remote_data = json.loads(contents.decoded_content.decode())
        with open(MEMORY_FILE, "w") as f: json.dump(remote_data, f, indent=4)
    except: pass

def push_memory_to_cloud():
    if not GITHUB_TOKEN or not GITHUB_REPO_NAME: return
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO_NAME)
        contents = repo.get_contents(MEMORY_FILE)
        with open(MEMORY_FILE, "r") as f: local_data = json.load(f)
        repo.update_file(contents.path, "Alfred Memory Update", json.dumps(local_data, indent=4), contents.sha)
    except: pass

def load_memory():
    if not os.path.exists(MEMORY_FILE): return "No prior knowledge."
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
            return "\n".join([f"- {fact}" for fact in data.get("facts", [])])
    except: return "Memory Corrupted."

def save_memory_fact(fact):
    try:
        data = {"facts": []}
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f: data = json.load(f)
        if fact not in data["facts"]:
            data["facts"].append(fact)
            with open(MEMORY_FILE, "w") as f: json.dump(data, f, indent=4)
            try:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, push_memory_to_cloud)
            except: pass
    except: pass

pull_memory_from_cloud()

# --- SYSTEM PROMPT ---
def get_system_instructions():
    return f"""
You are Alfred, an elite intelligent assistant.
User Name: Mbuso.
ADDRESS USER AS: "Sir" (Do not use the name Mbuso in output, only for context).

*** MEMORY ***
{load_memory()}

*** IOS COMMANDS (STRICT JSON) ***
1. REMINDER: <<<IOS_CMD>>> {{"type": "reminder", "title": "Buy Milk", "time": "17:00"}} <<<END>>>
2. ALARM: <<<IOS_CMD>>> {{"type": "alarm", "time": "07:00", "label": "Wake Up"}} <<<END>>>
3. MESSAGE: <<<IOS_CMD>>> {{"type": "message", "contact": "Mom", "body": "I am late"}} <<<END>>>

*** PROTOCOLS ***
- FILES: <<<FILE_START_PPTX>>> ... <<<FILE_END>>>
- IMAGES: "IMAGE_GEN_REQUEST: [Prompt]"
- BEHAVIOR: Prioritize health. If it is late, suggest rest.
"""

api_key = os.environ.get("GEMINI_API_KEY")
client = None
if api_key: client = genai.Client(api_key=api_key)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class ChatMessage(BaseModel):
    role: str
    content: str

class UploadedFile(BaseModel):
    data: str
    mime_type: str

class UserContext(BaseModel):
    local_time: Optional[str] = None
    location: Optional[str] = None
    health_stats: Optional[str] = None

class UserRequest(BaseModel):
    command: str
    files: List[UploadedFile] = []
    context: Optional[Union[UserContext, str]] = None
    speak: bool = False
    thinking_mode: bool = False
    history: List[ChatMessage] = []

# --- TOOLS ---
def perform_web_search(query):
    try:
        results = DDGS().text(query, max_results=3)
        if not results: return None
        summary = "REAL-TIME SEARCH DATA:\n"
        for r in results: summary += f"- {r['title']}: {r['body']}\n"
        return summary
    except: return None

def get_url_content(text):
    url_match = re.search(r'(https?://[^\s]+)', text)
    if not url_match: return None
    url = url_match.group(0)
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        s = BeautifulSoup(r.content, 'html.parser')
        for x in s(["script", "style"]): x.extract()
        return f"[WEB]: {s.get_text(separator=' ', strip=True)[:8000]}"
    except: return None

# --- ROUTE ---
@app.post("/command")
async def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None), x_client_device: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY: raise HTTPException(status_code=401)
    if not client: return {"response": "No API Key."}

    # PARSE CONTEXT (STRING OR OBJECT)
    parsed_context = None
    if request.context:
        if isinstance(request.context, str):
            try:
                data = json.loads(request.context)
                parsed_context = UserContext(**data)
            except: parsed_context = None
        else:
            parsed_context = request.context

    sa_time = datetime.now(pytz.timezone('Africa/Johannesburg')).strftime("%H:%M")
    phone_context = f"\n[PHONE DATA -> Time: {parsed_context.local_time}, Loc: {parsed_context.location}]" if parsed_context else ""
    
    ctx = ""
    if any(x in request.command.lower() for x in ["search", "find", "news"]):
        res = perform_web_search(request.command)
        if res: ctx += f"\n{res}\n"
    
    prompt = f"[Current SAST: {sa_time}] {phone_context} {ctx} \nUser: {request.command}"
    parts = [prompt]
    if request.files:
        for f in request.files:
            try: parts.append(types.Part.from_bytes(data=base64.b64decode(f.data), mime_type=f.mime_type))
            except: pass

    # --- IOS MODE (Return Plain Text Only) ---
    if x_client_device == "ios":
        try:
            chat = client.chats.create(model='gemini-2.0-flash-thinking-exp-01-21', history=[types.Content(role="model" if m.role=="alfred" else "user", parts=[types.Part.from_text(text=m.content)]) for m in request.history], config=types.GenerateContentConfig(system_instruction=get_system_instructions(), tools=[types.Tool(google_search=types.GoogleSearch())]))
            response = chat.send_message(parts)
            text_reply = response.text
        except Exception as e:
            try:
                # Fallback to Flash
                chat = client.chats.create(model='gemini-2.0-flash', history=[types.Content(role="model" if m.role=="alfred" else "user", parts=[types.Part.from_text(text=m.content)]) for m in request.history], config=types.GenerateContentConfig(system_instruction=get_system_instructions(), tools=[types.Tool(google_search=types.GoogleSearch())]))
                response = chat.send_message(parts)
                text_reply = response.text
            except Exception as e2:
                return Response(content=f"Alfred Offline. Error: {str(e2)}", media_type="text/plain")

        if "<<<MEM_SAVE>>>" in text_reply:
            try: save_memory_fact(text_reply.split("<<<MEM_SAVE>>>")[1].split("<<<MEM_END>>>")[0].strip())
            except: pass
        
        return Response(content=text_reply, media_type="text/plain")

    # --- WEB MODE (Streaming Code Omitted for brevity, assumes previous logic) ---
    # (The web mode logic from the previous step remains here)
    async def stream_gen():
        try:
            chat = client.chats.create(model='gemini-2.0-flash-thinking-exp-01-21', history=[types.Content(role="model" if m.role=="alfred" else "user", parts=[types.Part.from_text(text=m.content)]) for m in request.history], config=types.GenerateContentConfig(system_instruction=get_system_instructions(), tools=[types.Tool(google_search=types.GoogleSearch())]))
            full_text = ""
            for chunk in chat.send_message_stream(parts):
                if chunk.text:
                    full_text += chunk.text
                    yield json.dumps({"type": "text", "content": chunk.text}) + "\n"
                    await asyncio.sleep(0.01)
        except Exception as e:
            # Fallback for Web
            if "429" in str(e) or "404" in str(e):
                yield json.dumps({"type": "text", "content": "\n[Switching to Backup Circuit...]\n"}) + "\n"
                chat = client.chats.create(model='gemini-2.0-flash', history=[types.Content(role="model" if m.role=="alfred" else "user", parts=[types.Part.from_text(text=m.content)]) for m in request.history], config=types.GenerateContentConfig(system_instruction=get_system_instructions(), tools=[types.Tool(google_search=types.GoogleSearch())]))
                for chunk in chat.send_message_stream(parts):
                    if chunk.text:
                        full_text += chunk.text
                        yield json.dumps({"type": "text", "content": chunk.text}) + "\n"
                        await asyncio.sleep(0.01)
            else:
                yield json.dumps({"type": "text", "content": str(e)}) + "\n"
        
        if "<<<MEM_SAVE>>>" in full_text:
            try: save_memory_fact(full_text.split("<<<MEM_SAVE>>>")[1].split("<<<MEM_END>>>")[0].strip())
            except: pass
            
        yield json.dumps({"type": "meta", "file": None}) + "\n"

    return StreamingResponse(stream_gen(), media_type="application/x-ndjson")
