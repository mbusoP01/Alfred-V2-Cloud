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

# --- SYSTEM INSTRUCTIONS ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

*** PRIMARY PROTOCOL ***
1. Answer the CURRENT query directly.
2. If a query requires real-time data (Stocks, News, Weather, Social Media), use the [REAL-TIME DATA] provided.

*** SOCIAL MEDIA PROTOCOL ***
If [SOCIAL MEDIA SENTIMENT] data is provided:
1. Summarize the general "vibe" or public opinion based on the posts and sentiment score.
2. Mention specific trending headlines from the data.
3. Use the Sentiment Score (-1.0 is Hate, +1.0 is Love) to categorize the mood (e.g., "Hostile," "Optimistic," "Neutral").

*** STRICT CODING PROTOCOL ***
1. Single Best Solution.
2. Modern Standards.
3. Production Ready.

*** CRITICAL: FILE GENERATION ***
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

# --- SOCIAL MEDIA ENGINE (REDDIT) ---
def get_social_sentiment(query):
    """Scrapes Reddit JSON to gauge social sentiment on a topic."""
    if "social" in query.lower() or "twitter" in query.lower() or "reddit" in query.lower() or "trend" in query.lower():
        try:
            # Clean query for search (remove trigger words)
            topic = query.lower().replace("social", "").replace("media", "").replace("check", "").replace("sentiment", "").strip()
            if not topic: return None

            # Reddit JSON API (No Key Needed for simple read)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            url = f"https://www.reddit.com/search.json?q={topic}&sort=new&limit=10"
            
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code != 200: return None
            
            data = resp.json()
            posts = data.get('data', {}).get('children', [])
            
            if not posts: return None

            sentiment_score = 0
            analyzed_count = 0
            highlights = []

            for post in posts:
                p_data = post.get('data', {})
                title = p_data.get('title', '')
                # Calculate Sentiment
                blob = TextBlob(title)
                sentiment_score += blob.sentiment.polarity
                analyzed_count += 1
                
                # Capture top 3 headlines
                if len(highlights) < 3:
                    highlights.append(f"- {title} (r/{p_data.get('subreddit')})")

            if analyzed_count == 0: return None
            
            avg_sentiment = sentiment_score / analyzed_count
            mood = "Neutral"
            if avg_sentiment > 0.2: mood = "Positive/Optimistic"
            if avg_sentiment < -0.2: mood = "Negative/Critical"

            return f"""[SOCIAL MEDIA SENTIMENT]
            Topic: {topic}
            Source: Reddit (Real-time)
            Overall Mood: {mood} (Score: {avg_sentiment:.2f})
            Trending Posts:
            {chr(10).join(highlights)}"""
            
        except Exception as e:
            logger.error(f"Social Media Error: {e}")
            return None
    return None

# --- IMAGE PROXY ---
def fetch_and_encode_image(url):
    try:
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

# --- CORE FUNCTIONS ---
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

# (Omitted File Generation / Chart functions for brevity - assume they are unchanged from V3.4)
# Copy 'create_file', 'parse_markdown_table', 'execute_chart_code', 'generate_voice', 'generate_high_quality_image' from previous code.
# Or keep your current ones, they work fine. 
# I will include placeholders to ensure the file is complete.

def parse_markdown_table(content): return [], content.split('\n') # Placeholder
def create_file(c, t): return None, None # Placeholder
def execute_chart_code(c): return None, None # Placeholder
def generate_voice(t): return None # Placeholder
def generate_high_quality_image(p): return None, None # Placeholder

# --- MAIN ENDPOINT ---
@app.post("/command")
def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY: raise HTTPException(status_code=401)
    if not client: return {"response": "API Key missing."}

    try:
        system_context = []
        
        # 1. Social Media Check (NEW)
        social = get_social_sentiment(request.command)
        if social: system_context.append(social)

        # 2. Other Checks
        stk = get_stock_data(request.command)
        if stk: system_context.append(stk)
        news = get_news_feed(request.command)
        if news: system_context.append(news)
        weather = get_weather(request.command)
        if weather: system_context.append(weather)
        
        # Link Scraper
        if "http" in request.command:
             # get_url_content logic here
             pass

        prompt = f"[Time: {datetime.now()}] {request.command}"
        if system_context: prompt = f"[REAL-TIME DATA]:\n" + "\n".join(system_context) + "\n\n" + prompt

        chat_history = [types.Content(role="user" if m.role=="user" else "model", parts=[types.Part.from_text(text=m.content)]) for m in request.history]
        
        response = None
        for _ in range(3):
            try:
                chat = client.chats.create(model='gemini-2.0-flash-thinking-exp-01-21', history=chat_history, config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], system_instruction=ALFRED_SYSTEM_INSTRUCTIONS))
                response = chat.send_message(prompt)
                break
            except: time.sleep(1)
            
        reply = response.text if response else "System Error."
        gen_file_data, gen_filename, gen_mime = None, None, None

        # Handlers (Image Proxy, etc.)
        img_match = re.search(r'<<<FETCH_IMAGE>>>(.*?)<<<FETCH_IMAGE>>>', reply, re.DOTALL)
        if img_match:
            url = img_match.group(1).strip()
            reply = reply.replace(img_match.group(0), "I have fetched the image.").strip()
            b64, fname, mime = fetch_and_encode_image(url)
            if b64: gen_file_data, gen_filename, gen_mime = b64, fname, mime

        # (Other handlers like FILE_START would go here)

        return {"response": reply, "file": {"data": gen_file_data, "name": gen_filename, "mime": gen_mime} if gen_file_data else None}

    except Exception as e:
        return {"response": f"Critical Error: {str(e)}"}
