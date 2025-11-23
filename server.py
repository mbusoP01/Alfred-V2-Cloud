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
from fpdf.fonts import FontFace
from docx import Document
from pptx import Presentation
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

# --- KEYS ---
SERVER_SECRET_KEY = "Mbuso.08@"
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

PROTOCOL:
1. Answer the CURRENT query directly.
2. IF A LINK IS PROVIDED: The server has read it. Summarize or analyze the 'Link Content'.
3. IF A FILE IS UPLOADED: Analyze it.
4. ERRORS: If you encounter an error (like a YouTube link you can't read), explain EXACTLY why to the user.

*** CRITICAL: FILE GENERATION ***
If user asks to create/generate a file:
1. Do NOT chat.
2. Wrap content in: <<<FILE_START>>> content <<<FILE_END>>>.
3. For PDF TABLES: Use standard Markdown tables. I will format them beautifully.

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

# --- RESEARCHER ENGINE (WEB SCRAPER) ---
def get_url_content(text):
    url_match = re.search(r'(https?://[^\s]+)', text)
    if not url_match: return None
    url = url_match.group(0)
    print(f"Scraping URL: {url}")
    
    try:
        # YOUTUBE LOGIC
        if "youtube.com" in url or "youtu.be" in url:
            video_id = None
            if "v=" in url: video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be" in url: video_id = url.split("/")[-1]
            
            if video_id:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    full_text = " ".join([t['text'] for t in transcript])
                    return f"[YOUTUBE TRANSCRIPT]: {full_text[:15000]}"
                except Exception as e:
                    # Specific YouTube Errors
                    if "TranscriptsDisabled" in str(e):
                        return "[ERROR]: This video has subtitles disabled. I cannot summarize it."
                    elif "NoTranscriptFound" in str(e):
                        return "[ERROR]: No English transcript found for this video."
                    else:
                        return f"[ERROR READING YOUTUBE]: {str(e)}"
        
        # STANDARD WEB LOGIC
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]): script.extract()
        text_content = soup.get_text(separator=' ', strip=True)
        return f"[WEBSITE CONTENT]: {text_content[:10000]}"

    except Exception as e:
        return f"[ERROR READING LINK]: {str(e)}"

# --- IMAGE GEN ---
def generate_high_quality_image(prompt):
    if not HUGGINGFACE_API_KEY: return None, None
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    models = [
        "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev",
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large",
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    ]
    for model_url in models:
        try:
            response = requests.post(model_url, headers=headers, json=payload)
            if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
                image_b64 = base64.b64encode(response.content).decode('utf-8')
                filename = f"gen_{datetime.now().strftime('%H%M%S')}.jpg"
                return image_b64, filename
            elif response.status_code == 503:
                time.sleep(1)
                continue
        except: continue
    return None, None

# --- FILE FACTORY ---
def create_file(content, file_type):
    buffer = io.BytesIO()
    timestamp = datetime.now().strftime('%H%M%S')
    filename = f"alfred_doc_{timestamp}"
    
    try:
        if file_type == "pdf":
            # UPDATED PDF LOGIC WITH TABLES
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            
            lines = content.split('\n')
            
            # Simple parser to detect tables vs text
            table_buffer = []
            in_table = False
            
            for line in lines:
                # Basic Markdown Table Detection
                if "|" in line and len(line.strip()) > 2:
                    if not in_table:
                        in_table = True
                        pdf.ln(5) # Space before table
                    
                    # Clean markdown row
                    row_data = [cell.strip() for cell in line.split('|') if cell.strip()]
                    # Ignore separator lines like |---|---|
                    if "---" not in line:
                        table_buffer.append(row_data)
                else:
                    # If we were in a table and now hit text, render the table
                    if in_table and table_buffer:
                        try:
                            with pdf.table() as table:
                                for row in table_buffer:
                                    r = table.row()
                                    for item in row:
                                        r.cell(item)
                        except Exception as e:
                            pdf.multi_cell(0, 10, "Error rendering table: " + str(e))
                        
                        table_buffer = []
                        in_table = False
                        pdf.ln(5) # Space after table

                    # Render normal text
                    if line.strip():
                        # Handle headers crudely
                        if line.startswith("#"):
                            pdf.set_font("Helvetica", style="B", size=14)
                            pdf.cell(0, 10, line.replace("#", "").strip(), new_x="LMARGIN", new_y="NEXT")
                            pdf.set_font("Helvetica", size=12)
                        else:
                            safe_text = line.encode('latin-1', 'replace').decode('latin-1')
                            pdf.multi_cell(0, 7, safe_text)
            
            # Flush final table if exists
            if in_table and table_buffer:
                 with pdf.table() as table:
                    for row in table_buffer:
                        r = table.row()
                        for item in row:
                            r.cell(item)

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
    
    return None, None

# --- CHART ENGINE ---
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

# --- VOICE ---
def generate_voice(text):
    if not ELEVENLABS_API_KEY: return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{BRIAN_VOICE_ID}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    clean_text = re.sub(r'IMAGE_GEN_REQUEST:.*', 'Generating image.', text)
    clean_text = re.sub(r'<<<.*?>>>', 'Content generated.', clean_text, flags=re.DOTALL)
    clean_text = clean_text.replace("*", "").replace("#", "").replace("`", "")
    if not clean_text.strip(): return None
    try:
        res = requests.post(url, json={"text": clean_text[:1000], "model_id": "eleven_monolingual_v1"}, headers=headers)
        if res.status_code == 200: return base64.b64encode(res.content).decode('utf-8')
    except: pass
    return None

@app.get("/")
def home():
    return {"status": "Alfred V2.1 Online (Cloud)"}

@app.post("/command")
def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY:
        raise HTTPException(status_code=401, detail="ACCESS DENIED")

    if not client: return {"response": "Sir, connection severed (No API Key)."}

    try:
        sa_timezone = pytz.timezone('Africa/Johannesburg')
        now = datetime.now(sa_timezone).strftime("%A, %B %d, %Y at %I:%M %p (SAST)")
        
        requested_type = None
        lower_cmd = request.command.lower()
        if "pdf" in lower_cmd: requested_type = "pdf"
        elif "word" in lower_cmd or "docx" in lower_cmd: requested_type = "docx"
        elif "presentation" in lower_cmd or "pptx" in lower_cmd: requested_type = "pptx"
        elif "text file" in lower_cmd or "txt" in lower_cmd: requested_type = "txt"

        selected_model = 'gemini-2.0-flash-thinking-exp-01-21' if request.thinking_mode else 'gemini-2.0-flash'

        chat_history = []
        for msg in request.history:
            role = "model" if msg.role == "alfred" else "user"
            chat_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        # --- SCRAPE LINK ---
        scraped_content = get_url_content(request.command)
        if scraped_content:
            prompt_text = f"[System: User provided a link. Content: {scraped_content}]\n\nUser Query: {request.command}"
        else:
            prompt_text = f"[Current SAST Time: {now}] {request.command}"

        current_parts = [prompt_text]
        if request.file_data:
            try:
                file_bytes = base64.b64decode(request.file_data)
                current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=request.mime_type))
            except: pass

        chat_session = client.chats.create(
            model=selected_model,
            history=chat_history,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
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
        
        gen_file_data = None
        gen_filename = None

        # --- 1. FILE GEN FIX ---
        if "<<<FILE_START>>>" in reply:
            try:
                content = reply.split("<<<FILE_START>>>")[1].split("<<<FILE_END>>>")[0].strip()
                # Remove code block from chat for cleaner UI
                reply = "I have manufactured the document for you, Sir."
                if not requested_type: requested_type = "txt"
                
                gen_file_data, gen_filename = create_file(content, requested_type)
                if gen_file_data:
                    reply = f"I have successfully created the {requested_type.upper()} file."
                else:
                    reply = "I attempted to create the file, but an internal formatting error occurred."
            except Exception as e:
                reply = f"I prepared the content, but the file assembly failed. Error: {str(e)}"

        # --- 2. IMAGE GEN ---
        elif "IMAGE_GEN_REQUEST:" in reply:
            image_prompt = reply.split("IMAGE_GEN_REQUEST:")[1].strip()
            reply = "I am rendering the image, Sir."
            gen_file_data, gen_filename = generate_high_quality_image(image_prompt)
            if not gen_file_data: reply = "Visual engine is busy. Please retry."

        # --- 3. CHART GEN ---
        elif "<<<CHART_START>>>" in reply:
            try:
                code = reply.split("<<<CHART_START>>>")[1].split("<<<CHART_END>>>")[0].strip()
                reply = "I have visualized the data for you, Sir."
                gen_file_data, gen_filename = execute_chart_code(code)
            except: pass

        audio_data = generate_voice(reply) if request.speak else None

    except Exception as e:
        # RETURN ACTUAL ERROR TO USER
        reply = f"SYSTEM ERROR: {str(e)}"
        audio_data, gen_file_data, gen_filename = None, None, None

    return {"response": reply, "audio": audio_data, "file": {"data": gen_file_data, "name": gen_filename} if gen_file_data else None}
