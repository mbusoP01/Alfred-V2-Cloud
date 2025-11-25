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
from github import Github
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
from docx.shared import Pt, RGBColor as DocxColor
from pptx import Presentation
from pptx.util import Inches, Pt as PptxPt
from pptx.dml.color import RGBColor as PptxColor
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from duckduckgo_search import DDGS

# --- KEYS ---
SERVER_SECRET_KEY = "Mbuso.08@"
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

# --- CLOUD MEMORY CONFIG ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO_NAME = os.environ.get("GITHUB_REPO")
MEMORY_FILE = "alfred_memory.json"

# --- CLOUD SYNAPSE FUNCTIONS ---
def pull_memory_from_cloud():
    if not GITHUB_TOKEN or not GITHUB_REPO_NAME: return
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO_NAME)
        contents = repo.get_contents(MEMORY_FILE)
        remote_data = json.loads(contents.decoded_content.decode())
        with open(MEMORY_FILE, "w") as f: json.dump(remote_data, f, indent=4)
        print("Memory Synced from Cloud.")
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

# --- MEMORY LOGIC ---
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

# --- INITIALIZE ---
pull_memory_from_cloud()

# --- SYSTEM PROMPT ---
def get_system_instructions():
    return f"""
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

*** MEMORY ***
{load_memory()}

PROTOCOL:
1. SEARCH: Use real-time data if provided.
2. MEMORY: Save new facts: <<<MEM_SAVE>>> fact <<<MEM_END>>>.
3. VISUALS:
   - Charts: <<<CHART_START>>> JSON <<<CHART_END>>>
   - Files: <<<FILE_START>>> Content <<<FILE_END>>>
   - Images: "IMAGE_GEN_REQUEST: Prompt"

*** PRESENTATION STYLE ***
When creating files, structure your content clearly with "Title: X" and bullet points so I can format them beautifully.
"""

api_key = os.environ.get("GEMINI_API_KEY")
client = None
if api_key: client = genai.Client(api_key=api_key)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

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

# --- 1. SEARCH ---
def perform_web_search(query):
    try:
        results = DDGS().text(query, max_results=3)
        if not results: return None
        summary = "REAL-TIME SEARCH DATA:\n"
        for r in results: summary += f"- {r['title']}: {r['body']}\n"
        return summary
    except: return None

# --- 2. SCRAPE ---
def get_url_content(text):
    url_match = re.search(r'(https?://[^\s]+)', text)
    if not url_match: return None
    url = url_match.group(0)
    try:
        if "youtube.com" in url or "youtu.be" in url:
            vid = url.split("v=")[1].split("&")[0] if "v=" in url else url.split("/")[-1]
            t = YouTubeTranscriptApi.get_transcript(vid)
            return f"[YOUTUBE]: {' '.join([x['text'] for x in t])[:8000]}"
        h = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=h, timeout=10)
        s = BeautifulSoup(r.content, 'html.parser')
        for x in s(["script", "style"]): x.extract()
        return f"[WEB]: {s.get_text(separator=' ', strip=True)[:8000]}"
    except: return None

# --- 3. VOICE ---
async def generate_voice_dual(text):
    clean = re.sub(r'<<<.*?>>>', '', text).replace('IMAGE_GEN_REQUEST:', '')
    clean = re.sub(r'http\S+', 'a link', clean)
    if not clean.strip(): return None
    if ELEVENLABS_API_KEY:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{BRIAN_VOICE_ID}"
            headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
            res = requests.post(url, json={"text": clean[:1000], "model_id": "eleven_monolingual_v1"}, headers=headers)
            if res.status_code == 200: return base64.b64encode(res.content).decode('utf-8')
        except: pass
    try:
        communicate = edge_tts.Communicate(clean, "en-GB-RyanNeural")
        filename = f"voice_{int(time.time())}.mp3"
        await communicate.save(filename)
        with open(filename, "rb") as f: audio_bytes = f.read()
        os.remove(filename)
        return base64.b64encode(audio_bytes).decode('utf-8')
    except: return None

# --- 4. IMAGE ---
def generate_image(prompt):
    if not HUGGINGFACE_API_KEY: return None, None
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    models = ["https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"]
    for m in models:
        try:
            r = requests.post(m, headers=headers, json={"inputs": prompt}, timeout=10)
            if r.status_code == 200: return base64.b64encode(r.content).decode('utf-8'), "gen.jpg"
        except: continue
    return None, None

# --- 5. THE ARCHITECT (FILE GEN) ---
def create_file(content, ftype):
    try:
        buf = io.BytesIO()
        fname = f"Alfred_Export_{int(time.time())}"
        
        if ftype == "pdf":
            p = FPDF(); p.add_page(); p.set_font("Arial", size=12)
            # Add Header
            p.set_font("Arial", 'B', 16); p.cell(0, 10, "Alfred Intelligence Report", 0, 1, 'C'); p.ln(10)
            p.set_font("Arial", size=12)
            p.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1'))
            return base64.b64encode(p.output(dest='S').encode('latin-1')).decode('utf-8'), f"{fname}.pdf"
        
        elif ftype == "docx":
            d = Document()
            # Title
            title = d.add_heading('Alfred Intelligence Report', 0)
            title.alignment = 1 # Center
            d.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            d.add_paragraph("-" * 50)
            d.add_paragraph(content)
            d.save(buf)
            return base64.b64encode(buf.getvalue()).decode('utf-8'), f"{fname}.docx"
        
        elif ftype == "pptx":
            prs = Presentation()
            # Slide 1: Title
            slide1 = prs.slides.add_slide(prs.slide_layouts[0]) # Title Slide
            title = slide1.shapes.title
            subtitle = slide1.placeholders[1]
            title.text = "Alfred Intelligence Briefing"
            subtitle.text = f"Prepared for Mbuso (Sir)\n{datetime.now().strftime('%Y-%m-%d')}"
            
            # Dark Mode Styling (Manual)
            background = slide1.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = PptxColor(20, 20, 20) # Dark BG
            title.text_frame.paragraphs[0].font.color.rgb = PptxColor(10, 132, 255) # Blue Accent
            subtitle.text_frame.paragraphs[0].font.color.rgb = PptxColor(200, 200, 200) # Grey Text

            # Slide 2: Content
            slide2 = prs.slides.add_slide(prs.slide_layouts[1]) # Title + Content
            # Dark Mode Slide 2
            background = slide2.background; fill = background.fill; fill.solid(); fill.fore_color.rgb = PptxColor(20, 20, 20)
            
            # Content Parsing (Simple Split)
            lines = content.split('\n')
            header_text = lines[0] if lines else "Details"
            body_text = "\n".join(lines[1:]) if len(lines) > 1 else content

            title_shape = slide2.shapes.title
            title_shape.text = header_text[:50] # Limit title length
            title_shape.text_frame.paragraphs[0].font.color.rgb = PptxColor(10, 132, 255)
            
            body_shape = slide2.placeholders[1]
            body_shape.text = body_text
            for paragraph in body_shape.text_frame.paragraphs:
                paragraph.font.color.rgb = PptxColor(255, 255, 255) # White body text
                paragraph.font.size = PptxPt(18)

            prs.save(buf)
            return base64.b64encode(buf.getvalue()).decode('utf-8'), f"{fname}.pptx"
            
        elif ftype == "txt":
            return base64.b64encode(content.encode('utf-8')).decode('utf-8'), f"{fname}.txt"
            
    except Exception as e:
        print(f"File Gen Error: {e}")
        return None, None

# --- CHART ---
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

@app.post("/command")
async def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY: raise HTTPException(status_code=401)
    if not client: return {"response": "System Failure: No API Key."}

    try:
        sa_time = datetime.now(pytz.timezone('Africa/Johannesburg')).strftime("%H:%M")
        hist = [types.Content(role="model" if m.role=="alfred" else "user", parts=[types.Part.from_text(text=m.content)]) for m in request.history]
        
        ctx = ""
        scraped = get_url_content(request.command)
        if scraped: ctx += f"\n{scraped}\n"
        elif any(x in request.command.lower() for x in ["search", "find", "news", "price", "weather", "graph", "chart"]):
            search_res = perform_web_search(request.command)
            if search_res: ctx += f"\n{search_res}\n"

        prompt = f"[Time: {sa_time}] {ctx} \nUser: {request.command}"
        parts = [prompt]
        if request.file_data: parts.append(types.Part.from_bytes(data=base64.b64decode(request.file_data), mime_type=request.mime_type))

        chat = client.chats.create(model='gemini-2.0-flash-thinking-exp-01-21' if request.thinking_mode else 'gemini-2.0-flash', history=hist, config=types.GenerateContentConfig(system_instruction=get_system_instructions(), tools=[types.Tool(google_search=types.GoogleSearch())]))
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: chat.send_message(parts).text)
        
        audio, gen_file, gen_name, chart_data = None, None, None, None
        
        if "<<<MEM_SAVE>>>" in response:
            try:
                fact = response.split("<<<MEM_SAVE>>>")[1].split("<<<MEM_END>>>")[0].strip()
                save_memory_fact(fact)
                response = re.sub(r'<<<MEM_SAVE>>>.*?<<<MEM_END>>>', '', response, flags=re.DOTALL).strip()
            except: pass

        if "<<<CHART_START>>>" in response:
            try:
                json_str = response.split("<<<CHART_START>>>")[1].split("<<<CHART_END>>>")[0].strip()
                chart_data = json.loads(json_str)
                response = re.sub(r'<<<CHART_START>>>.*?<<<CHART_END>>>', '', response, flags=re.DOTALL).strip()
                if not response: response = "Here is the visual data, Sir."
            except: pass

        if "<<<FILE_START>>>" in response:
            try:
                cnt = response.split("<<<FILE_START>>>")[1].split("<<<FILE_END>>>")[0].strip()
                ftype = "docx" if "word" in request.command.lower() else "txt"
                if "presentation" in request.command.lower() or "ppt" in request.command.lower(): ftype = "pptx"
                elif "pdf" in request.command.lower(): ftype = "pdf"
                
                gen_file, gen_name = create_file(cnt, ftype)
                response = "Document ready, Sir."
            except: pass
            
        elif "IMAGE_GEN_REQUEST:" in response:
            p = response.split("IMAGE_GEN_REQUEST:")[1].strip()
            response = "Rendering image..."
            gen_file, gen_name = generate_image(p)

        if request.speak: audio = await generate_voice_dual(response)

    except Exception as e:
        print(f"Error: {e}")
        response = f"System Error: {str(e)}"

    return {"response": response, "audio": audio, "file": {"data": gen_file, "name": gen_name} if gen_file else None, "chart_data": chart_data}
