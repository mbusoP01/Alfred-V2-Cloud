# server.py
import os
import base64
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

# --- SYSTEM IDENTITY ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

PROTOCOL:
1. USE REAL-WORLD DATA: If 'SEARCH_DATA' is provided in the context, YOU MUST USE IT to answer.
2. CITATIONS: If you use the search data, mention the source briefly.
3. TONE: Professional, concise, loyal, and slightly witty.

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

# --- 1. RESEARCHER ENGINE (ACTIVE SEARCH) ---
def perform_web_search(query):
    """Active search using DuckDuckGo"""
    print(f"Searching Web for: {query}")
    try:
        results = DDGS().text(query, max_results=4)
        if not results: return None
        
        summary = "REAL-TIME SEARCH DATA FOUND:\n"
        for r in results:
            summary += f"- Title: {r['title']}\n  Link: {r['href']}\n  Snippet: {r['body']}\n\n"
        return summary
    except Exception as e:
        print(f"Search Error: {e}")
        return None

# --- 2. LINK READER (PASSIVE SCRAPING) ---
def get_url_content(text):
    """Reads specific links if provided"""
    url_match = re.search(r'(https?://[^\s]+)', text)
    if not url_match: return None
    url = url_match.group(0)
    print(f"Scraping URL: {url}")
    
    try:
        # YouTube Logic
        if "youtube.com" in url or "youtu.be" in url:
            video_id = None
            if "v=" in url: video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be" in url: video_id = url.split("/")[-1]
            
            if video_id:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = " ".join([t['text'] for t in transcript])
                return f"[YOUTUBE TRANSCRIPT]: {full_text[:10000]}"
        
        # Standard Web Logic
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]): script.extract()
        text_content = soup.get_text(separator=' ', strip=True)
        return f"[WEBSITE CONTENT]: {text_content[:10000]}"

    except Exception as e:
        return f"[ERROR READING LINK]: {str(e)}"

# --- 3. IMAGE GEN ---
def generate_high_quality_image(prompt):
    if not HUGGINGFACE_API_KEY: return None, None
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    # Using Turbo/Fast models for better success rate
    models = [
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
        "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    ]
    for model_url in models:
        try:
            response = requests.post(model_url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
                image_b64 = base64.b64encode(response.content).decode('utf-8')
                filename = f"gen_{datetime.now().strftime('%H%M%S')}.jpg"
                return image_b64, filename
        except: continue
    return None, None

# --- 4. FILE FACTORY ---
def create_file(content, file_type):
    buffer = io.BytesIO()
    timestamp = datetime.now().strftime('%H%M%S')
    filename = f"alfred_doc_{timestamp}"
    try:
        if file_type == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            # Fix unicode issues in FPDF
            safe_text = content.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, safe_text)
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
    except Exception as e:
        print(f"File Gen Error: {e}")
    return None, None

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
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    # Cleanup text for speech
    clean_text = re.sub(r'IMAGE_GEN_REQUEST:.*', 'Generating image.', text)
    clean_text = re.sub(r'<<<.*?>>>', 'Content generated.', clean_text, flags=re.DOTALL)
    clean_text = clean_text.replace("*", "").replace("#", "").replace("`", "")
    
    if not clean_text.strip(): return None
    try:
        res = requests.post(url, json={"text": clean_text[:800], "model_id": "eleven_monolingual_v1"}, headers=headers)
        if res.status_code == 200: return base64.b64encode(res.content).decode('utf-8')
    except: pass
    return None

@app.get("/")
def home():
    return {"status": "Alfred Online (Phase 1: Knowledge Upgrade)"}

@app.post("/command")
def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY:
        raise HTTPException(status_code=401, detail="ACCESS DENIED")

    if not client: return {"response": "Sir, connection severed (No API Key)."}

    try:
        # Time setup
        sa_timezone = pytz.timezone('Africa/Johannesburg')
        now = datetime.now(sa_timezone).strftime("%A, %B %d, %Y at %I:%M %p (SAST)")
        
        # 1. Determine Input Type
        requested_type = None
        lower_cmd = request.command.lower()
        if "pdf" in lower_cmd: requested_type = "pdf"
        elif "word" in lower_cmd or "docx" in lower_cmd: requested_type = "docx"
        elif "presentation" in lower_cmd or "pptx" in lower_cmd: requested_type = "pptx"
        elif "text file" in lower_cmd or "txt" in lower_cmd: requested_type = "txt"

        # 2. Select Brain
        selected_model = 'gemini-2.0-flash-thinking-exp-01-21' if request.thinking_mode else 'gemini-2.0-flash'

        # 3. Construct History
        chat_history = []
        for msg in request.history:
            role = "model" if msg.role == "alfred" else "user"
            chat_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        # 4. GATHER CONTEXT (The Knowledge Phase)
        additional_context = ""
        
        # A. Check for direct URL
        scraped_content = get_url_content(request.command)
        if scraped_content:
            additional_context += f"\n{scraped_content}\n"
        
        # B. Check for Search Trigger (Auto-Research)
        # Trigger words: search, find, news, latest, price, who is, what is
        search_triggers = ["search", "find", "news", "latest", "price", "who is", "what is", "weather"]
        if not scraped_content and any(x in lower_cmd for x in search_triggers):
             search_results = perform_web_search(request.command)
             if search_results:
                 additional_context += f"\n{search_results}\n"

        # 5. Build Final Prompt
        prompt_text = f"[Current SAST Time: {now}]"
        if additional_context:
            prompt_text += f"{additional_context}\n\nUser Query: {request.command}"
        else:
            prompt_text += f" {request.command}"

        current_parts = [prompt_text]
        
        # Handle File Uploads (Vision)
        if request.file_data:
            try:
                file_bytes = base64.b64decode(request.file_data)
                current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=request.mime_type))
            except: pass

        # 6. Generate Response
        chat_session = client.chats.create(
            model=selected_model,
            history=chat_history,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())], # Native Google Search as backup
                system_instruction=ALFRED_SYSTEM_INSTRUCTIONS,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                ]
            )
        )
        
        response = chat_session.send_message(message=current_parts)
        reply = response.text
        
        # 7. Post-Processing (Files, Images, Charts)
        gen_file_data = None
        gen_filename = None

        # File Logic
        if "<<<FILE_START>>>" in reply:
            try:
                content = reply.split("<<<FILE_START>>>")[1].split("<<<FILE_END>>>")[0].strip()
                reply = "I have successfully manufactured the document for you, Sir."
                if not requested_type: requested_type = "txt"
                gen_file_data, gen_filename = create_file(content, requested_type)
            except: reply = "I prepared the content, but the file assembly failed."

        # Image Logic
        elif "IMAGE_GEN_REQUEST:" in reply:
            image_prompt = reply.split("IMAGE_GEN_REQUEST:")[1].strip()
            reply = "I am rendering the image, Sir."
            gen_file_data, gen_filename = generate_high_quality_image(image_prompt)
            if not gen_file_data: reply = "Visual engine is currently overloaded. Please retry."

        # Chart Logic
        elif "<<<CHART_START>>>" in reply:
            try:
                code = reply.split("<<<CHART_START>>>")[1].split("<<<CHART_END>>>")[0].strip()
                reply = "I have visualized the data for you, Sir."
                gen_file_data, gen_filename = execute_chart_code(code)
            except: pass

        # Voice Logic
        audio_data = generate_voice(reply) if request.speak else None

    except Exception as e:
        print(f"Server Error: {e}")
        reply = f"I encountered an internal error: {str(e)}"
        audio_data, gen_file_data, gen_filename = None, None, None

    return {"response": reply, "audio": audio_data, "file": {"data": gen_file_data, "name": gen_filename} if gen_file_data else None}
