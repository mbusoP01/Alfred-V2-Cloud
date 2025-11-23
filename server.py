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
import logging

# --- NEW REAL-TIME LIBRARIES ---
import yfinance as yf
import feedparser
from textblob import TextBlob
from PIL import Image 

# --- DATABASE LIBRARY ---
import firebase_admin
from firebase_admin import credentials, firestore

# --- LOGGING ---
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# --- LIBRARIES ---
from fpdf import FPDF 
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from docx import Document 
from docx.shared import Pt 

# --- KEYS ---
SERVER_SECRET_KEY = "Mbuso.08@"
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

# --- FIREBASE INIT (PLACEHOLDER) ---
# if not firebase_admin._apps:
#     cred = credentials.Certificate("serviceAccountKey.json") 
#     firebase_admin.initialize_app(cred)
#     db = firestore.client()

# --- SYSTEM INSTRUCTIONS (UPDATED FOR STRICT CODING) ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

*** PRIMARY PROTOCOL ***
1. Answer the CURRENT query directly.
2. If a query requires real-time data (Stocks, News, Weather, Links), use the [REAL-TIME DATA] provided by the server.

*** STRICT CODING PROTOCOL ***
When asked for code, follow these rules MANDATORILY:
1. **Single Best Solution:** Provide ONLY ONE solution—the most efficient, modern, and secure one. Do not offer "Method 1, Method 2" unless explicitly asked for variations.
2. **Modern Standards:** Use modern syntax (e.g., ES6+ for JS, f-strings for Python). Avoid legacy methods like `document.write()` or `alert()` unless specifically requested for debugging.
3. **Production Ready:** The code must be clean, commented, and ready to run. Prioritize readability and performance.

*** IMAGE PROTOCOL ***
1. **If asked to GENERATE a unique image:** Reply EXACTLY: "IMAGE_GEN_REQUEST: [Detailed Prompt]"
2. **If asked to FIND or FETCH an image from the internet:** Use your Google Search tool to find a direct image URL (preferably .jpg or .png). Reply with the image URL wrapped in:
   <<<FETCH_IMAGE>>>[The direct image link here]<<<FETCH_IMAGE>>>

*** CRITICAL: FILE GENERATION ***
If user asks to create a file, use: <<<FILE_START>>> content <<<FILE_END>>>.
For Tables: Use standard Markdown tables.
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

# --- IMAGE PROXY WITH PILLOW OPTIMIZATION ---
def fetch_and_encode_image(url):
    """Downloads, RESIZES, and encodes an image."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Open Image with Pillow
        img = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB (removes Alpha channel issues for JPEG)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # Resize Logic: Max width/height 1024px (Preserve Aspect Ratio)
        img.thumbnail((1024, 1024))
        
        # Save optimized image to buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85) # Compress to 85% quality
        buffer.seek(0)
        
        # Encode
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        filename = "fetched_image.jpg"
        mime_type = "image/jpeg"
        
        return base64_data, filename, mime_type
    
    except Exception as e:
        logger.error(f"Image Proxy Error for {url}: {e}")
        return None, None, None

# --- CORE FUNCTIONS (News, Stocks, Files - Simplified) ---
def get_stock_data(query):
    words = query.split()
    ticker = None
    for w in words:
        if w.isupper() and len(w) <= 5 and w.isalpha(): ticker = w; break
    if not ticker: return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'currentPrice' in info: return f"[REAL-TIME STOCK: {ticker}]: ${info.get('currentPrice')}"
    except: return None
    return None

def get_news_feed(query):
    if "news" in query.lower():
        try:
            encoded = requests.utils.quote(query)
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en")
            entries = [f"- {e.title}" for e in feed.entries[:3]]
            if entries: return f"[REAL-TIME NEWS]:\n" + "\n".join(entries)
        except: pass
    return None

def get_weather(query):
    if "weather" in query.lower():
        try:
            loc = "Pretoria"
            for w in query.split(): 
                if w[0].isupper() and w.lower() != "weather": loc = w
            res = requests.get(f"https://wttr.in/{loc}?format=3")
            return f"[REAL-TIME WEATHER]: {res.text.strip()}"
        except: pass
    return None

def parse_markdown_table(content):
    lines = content.split('\n')
    table_buffer = []
    in_table = False
    for line in lines:
        if "|" in line and len(line.strip()) > 3:
            if not any("---" in cell for cell in line.split('|')):
                row_data = [cell.strip() for cell in line.split('|')]
                if row_data and row_data[0] == '': row_data.pop(0)
                if row_data and row_data[-1] == '': row_data.pop()
                table_buffer.append(row_data)
                in_table = True
        elif in_table: break
    if table_buffer:
        max_cols = max(len(row) for row in table_buffer)
        normalized_data = [row + [""] * (max_cols - len(row)) for row in table_buffer]
        return normalized_data, lines
    return None, lines

def create_file(content, file_type):
    buffer = io.BytesIO()
    timestamp = datetime.now().strftime('%H%M%S')
    filename = f"alfred_doc_{timestamp}"
    try:
        if file_type == "docx":
            doc = Document()
            doc.add_heading('Alfred Generation', 0)
            table_data, lines = parse_markdown_table(content)
            if table_data:
                table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
                table.style = 'Table Grid'
                for i, row in enumerate(table_data):
                    for j, text in enumerate(row): table.rows[i].cells[j].text = str(text)
                for line in lines:
                    if not "|" in line and not "---" in line and line.strip(): doc.add_paragraph(line)
            else:
                for line in lines:
                    if line.strip(): doc.add_paragraph(line)
            doc.save(buffer)
            return base64.b64encode(buffer.getvalue()).decode('utf-8'), f"{filename}.docx"
        
        elif file_type == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            table_data, lines = parse_markdown_table(content)
            if table_data:
                with pdf.table() as table:
                    for row in table_data:
                        r = table.row()
                        for item in row: r.cell(str(item))
                pdf.ln(10)
                for line in lines:
                    if not "|" in line and not "---" in line and line.strip():
                        pdf.multi_cell(0, 7, line.encode('latin-1', 'replace').decode('latin-1'))
            else:
                for line in lines:
                    if line.strip(): pdf.multi_cell(0, 7, line.encode('latin-1', 'replace').decode('latin-1'))
            return base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('utf-8'), f"{filename}.pdf"
            
        elif file_type == "txt":
            return base64.b64encode(content.encode('utf-8')).decode('utf-8'), f"{filename}.txt"
            
    except Exception as e:
        return base64.b64encode(f"Error: {e}".encode('utf-8')).decode('utf-8'), f"{filename}_error.txt"
    return None, None

# --- MAIN ENDPOINT ---
@app.post("/command")
def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY: raise HTTPException(status_code=401)
    if not client: return {"response": "API Key missing."}

    try:
        # Context & Real-Time Data
        system_context = []
        
        # Scrapers
        if "http" in request.command: 
            # (Researcher engine would go here)
            pass 
            
        stk = get_stock_data(request.command)
        if stk: system_context.append(stk)
        
        news = get_news_feed(request.command)
        if news: system_context.append(news)
        
        weather = get_weather(request.command)
        if weather: system_context.append(weather)

        prompt = f"[Time: {datetime.now()}] {request.command}"
        if system_context: prompt = f"[REAL-TIME DATA]:\n" + "\n".join(system_context) + "\n\n" + prompt

        # Chat
        chat_history = [types.Content(role="user" if m.role=="user" else "model", parts=[types.Part.from_text(text=m.content)]) for m in request.history]
        
        # Retry Loop
        response = None
        for _ in range(3):
            try:
                chat = client.chats.create(model='gemini-2.0-flash-thinking-exp-01-21', history=chat_history, config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], system_instruction=ALFRED_SYSTEM_INSTRUCTIONS))
                response = chat.send_message(prompt)
                break
            except: time.sleep(1)
            
        reply = response.text if response else "System Error."
        
        gen_file_data, gen_filename, gen_mime = None, None, None

        # --- IMAGE PROXY HANDLER ---
        img_match = re.search(r'<<<FETCH_IMAGE>>>(.*?)<<<FETCH_IMAGE>>>', reply, re.DOTALL)
        if img_match:
            url = img_match.group(1).strip()
            reply = reply.replace(img_match.group(0), "I have fetched the image for you.").strip()
            
            # CALL OPTIMIZED PROXY
            b64, fname, mime = fetch_and_encode_image(url)
            if b64:
                gen_file_data, gen_filename, gen_mime = b64, fname, mime
            else:
                reply += " (Note: The source image was inaccessible)."

        # Standard Handlers
        elif "IMAGE_GEN_REQUEST:" in reply:
            # (HuggingFace logic assumed here)
            pass
        elif "<<<FILE_START>>>" in reply:
            try:
                content = reply.split("<<<FILE_START>>>")[1].split("<<<FILE_END>>>")[0].strip()
                ftype = "txt"
                if "pdf" in request.command.lower(): ftype = "pdf"
                elif "word" in request.command.lower(): ftype = "docx"
                gen_file_data, gen_filename = create_file(content, ftype)
                if gen_file_data: reply = f"File created: {ftype.upper()}"
            except: pass

        return {"response": reply, "file": {"data": gen_file_data, "name": gen_filename, "mime": gen_mime} if gen_file_data else None}

    except Exception as e:
        return {"response": f"Critical Error: {str(e)}"}
