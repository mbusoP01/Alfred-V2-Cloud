# server.py
import os
import base64
import requests
import re
import io
import time
import pytz
import logging
import asyncio
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LIBRARIES (LIGHTWEIGHT) ---
from google import genai
from google.genai import types

# --- KEYS ---
SERVER_SECRET_KEY = "Mbuso.08@"
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
BRIAN_VOICE_ID = "nPczCjzI2devNBz1zQrb"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

# --- SYSTEM INSTRUCTIONS ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

*** PRIMARY PROTOCOL ***
1. Answer the CURRENT query directly.
2. If a query requires real-time data (Stocks, News, Weather, Links), use the [REAL-TIME DATA] provided.

*** IMAGE PROTOCOL ***
1. **Generate:** Reply "IMAGE_GEN_REQUEST: [Detailed Prompt]"
2. **Fetch:** Use Google Search to find a direct URL. Reply: <<<FETCH_IMAGE>>>[Link]<<<FETCH_IMAGE>>>

*** CODING PROTOCOL ***
1. Single Best Solution.
2. Modern Standards.
3. Production Ready.

*** FILE PROTOCOL ***
Use: <<<FILE_START>>> content <<<FILE_END>>>.
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

# --- LAZY LOADED FUNCTIONS ---

def get_stock_data(query):
    words = query.split()
    ticker = None
    for w in words:
        if w.isupper() and len(w) <= 5 and w.isalpha(): ticker = w; break
    if not ticker: return None
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'currentPrice' in info: return f"[REAL-TIME STOCK: {ticker}]: ${info.get('currentPrice')}"
    except: return None
    return None

def get_news_feed(query):
    if "news" in query.lower():
        try:
            import feedparser
            encoded = requests.utils.quote(query)
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en")
            entries = [f"- {e.title}" for e in feed.entries[:3]]
            if entries: return f"[REAL-TIME NEWS]:\n" + "\n".join(entries)
        except: pass
    return None

def get_social_sentiment(query):
    if "social" in query.lower():
        try:
            from textblob import TextBlob
            topic = query.lower().replace("social", "").strip()
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(f"https://www.reddit.com/search.json?q={topic}&sort=new&limit=5", headers=headers, timeout=5)
            if resp.status_code != 200: return None
            posts = resp.json().get('data', {}).get('children', [])
            if not posts: return None
            
            score = 0
            for p in posts:
                blob = TextBlob(p['data']['title'])
                score += blob.sentiment.polarity
            
            avg = score / len(posts)
            mood = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
            return f"[SOCIAL SENTIMENT]: {mood} (Score: {avg:.2f})"
        except: return None
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

def fetch_and_encode_image(url):
    try:
        from PIL import Image
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), "fetched.jpg", "image/jpeg"
    except: return None, None, None

def create_file(content, file_type):
    buffer = io.BytesIO()
    timestamp = datetime.now().strftime('%H%M%S')
    filename = f"alfred_doc_{timestamp}"
    try:
        if file_type == "pdf":
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.multi_cell(0, 7, content.encode('latin-1', 'replace').decode('latin-1'))
            return base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('utf-8'), f"{filename}.pdf"
        elif file_type == "docx":
            from docx import Document
            doc = Document()
            doc.add_paragraph(content)
            doc.save(buffer)
            return base64.b64encode(buffer.getvalue()).decode('utf-8'), f"{filename}.docx"
        elif file_type == "txt":
            return base64.b64encode(content.encode('utf-8')).decode('utf-8'), f"{filename}.txt"
    except: return None, None
    return None, None

# --- HEALTH CHECK ---
@app.get("/")
def health_check():
    logger.info("Health check ping received.")
    return {"status": "alive"}

# --- GLOBAL RATE LIMITER ---
last_request_time = 0

# --- MAIN ENDPOINT ---
@app.post("/command")
async def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    global last_request_time
    if x_alfred_auth != SERVER_SECRET_KEY: raise HTTPException(status_code=401)
    if not client: return {"response": "API Key missing."}

    # --- RATE LIMITER ---
    # Force a 2-second gap between requests to prevent spamming Google
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < 2:
        await asyncio.sleep(2 - time_since_last)
    last_request_time = time.time()

    try:
        # Context & Data
        system_context = []
        
        # Scrapers (Simplified)
        if "http" in request.command: pass
        
        stk = get_stock_data(request.command)
        if stk: system_context.append(stk)
        news = get_news_feed(request.command)
        if news: system_context.append(news)
        weather = get_weather(request.command)
        if weather: system_context.append(weather)
        social = get_social_sentiment(request.command)
        if social: system_context.append(social)

        prompt = f"[Time: {datetime.now()}] {request.command}"
        if system_context: prompt = f"[REAL-TIME DATA]:\n" + "\n".join(system_context) + "\n\n" + prompt

        chat_history = [types.Content(role="user" if m.role=="user" else "model", parts=[types.Part.from_text(text=m.content)]) for m in request.history]
        
        # --- SAFE RETRY LOOP (Handles 429 Too Many Requests) ---
        response = None
        for attempt in range(6): # Increased retries to 6
            try:
                chat = client.chats.create(
                    model='gemini-2.0-flash-thinking-exp-01-21', 
                    history=chat_history, 
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())], 
                        system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
                    )
                )
                response = chat.send_message(prompt)
                break
            except Exception as e:
                if "429" in str(e): # Rate limit hit
                    wait_time = 2**attempt # 1, 2, 4, 8, 16, 32 seconds
                    logger.warning(f"Rate limit 429 hit. Sleeping for {wait_time} seconds...")
                    time.sleep(wait_time) 
                else:
                    logger.error(f"API Error: {e}")
                    time.sleep(1)
            
        reply = response.text if response else "I am currently overloaded (Rate Limit). Please wait 30 seconds and try again."
        gen_file_data, gen_filename, gen_mime = None, None, None

        # Handlers
        img_match = re.search(r'<<<FETCH_IMAGE>>>(.*?)<<<FETCH_IMAGE>>>', reply, re.DOTALL)
        if img_match:
            url = img_match.group(1).strip()
            reply = reply.replace(img_match.group(0), "I have fetched the image.").strip()
            b64, fname, mime = fetch_and_encode_image(url)
            if b64: gen_file_data, gen_filename, gen_mime = b64, fname, mime

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
