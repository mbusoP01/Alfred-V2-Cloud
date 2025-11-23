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

# --- SYSTEM INSTRUCTIONS (UPDATED FOR SYNTHESIS & BIAS) ---
ALFRED_SYSTEM_INSTRUCTIONS = """
You are Alfred, an elite intelligent assistant.
User: Mbuso (Sir). Location: South Africa.

*** REAL-TIME DATA PROTOCOL ***
1. The server may inject [REAL-TIME DATA] into your prompt (Stocks, News, Weather).
2. USE this data to answer the user. Do not hallucinate if data is provided.
3. If the user asks for "Sentiment", use the provided TextBlob score.

*** ADVANCED SYNTHESIS & NLP PROTOCOL ***
1. **Cross-Document Synthesis:** If multiple sources are provided (links, text), identify COMMON themes and CONFLICTING viewpoints.
2. **Bias Detection:** Actively scan sources for subjective language. If a source is biased, note it: "Note: Source A demonstrates a slight bias towards..."
3. **Knowledge Graph Thinking:** Before summarizing, mentally map entities and their relationships. Ensure your summary reflects the *cause-and-effect* connections, not just a list of facts.

*** CRITICAL: CODE GENERATION ***
1. Quality Priority: ALWAYS prioritize the most efficient, modern solution.
2. No Legacy Methods: Avoid alert() or document.write().

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

# --- REAL-TIME DATA ENGINE ---

def get_stock_data(query):
    """Fetches stock data using yfinance if a ticker is found."""
    # Simple regex to find tickers like $AAPL or TSLA (simplified)
    words = query.split()
    ticker = None
    for w in words:
        if w.isupper() and len(w) <= 5 and w.isalpha(): 
            # Simple heuristic: if it's ALL CAPS and short, try it as a ticker
            ticker = w
            break
    
    if not ticker: return None

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'currentPrice' in info:
            return f"[REAL-TIME STOCK: {ticker}]: Price: ${info.get('currentPrice')} | Day High: ${info.get('dayHigh')} | Summary: {info.get('longBusinessSummary')[:200]}..."
    except:
        return None
    return None

def get_news_feed(query):
    """Fetches Google News RSS if 'news' is requested."""
    if "news" in query.lower():
        try:
            # Search Google News RSS
            encoded_query = requests.utils.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            entries = []
            for entry in feed.entries[:5]:
                entries.append(f"- {entry.title} (Source: {entry.source.title})")
            
            if entries:
                return f"[REAL-TIME NEWS]:\n" + "\n".join(entries)
        except Exception as e:
            logger.error(f"News fetch failed: {e}")
    return None

def get_weather(query):
    """Scrapes wttr.in for weather."""
    if "weather" in query.lower():
        try:
            # Default to South Africa if no location specified
            location = "Pretoria" 
            for word in query.split():
                if word[0].isupper() and word.lower() != "weather":
                    location = word
            
            response = requests.get(f"https://wttr.in/{location}?format=3")
            return f"[REAL-TIME WEATHER]: {response.text.strip()}"
        except: pass
    return None

def analyze_sentiment(text):
    """Basic sentiment check."""
    if "sentiment" in text.lower():
        blob = TextBlob(text)
        return f"[SENTIMENT ANALYSIS]: Polarity: {blob.sentiment.polarity} (-1.0 to 1.0), Subjectivity: {blob.sentiment.subjectivity}"
    return None

# --- RESEARCHER ENGINE (UPDATED) ---
def get_url_content(text):
    url_match = re.search(r'(https?://[^\s]+)', text)
    if not url_match: return None
    url = url_match.group(0)
    logger.info(f"Scraping URL: {url}")
    
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
                except Exception:
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                        res = requests.get(url, headers=headers, timeout=5)
                        soup = BeautifulSoup(res.content, 'html.parser')
                        title = soup.title.string if soup.title else "Unknown Title"
                        return f"[YOUTUBE_FALLBACK_METADATA]: Title: {title} | URL: {url} | (Transcript Blocked - INITIATE DEEP SEARCH)"
                    except Exception:
                        return f"[YOUTUBE ERROR]: Video inaccessible."

        # STANDARD WEB LOGIC
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]): script.extract()
        text_content = soup.get_text(separator=' ', strip=True)
        return f"[WEBSITE CONTENT]: {text_content[:10000]}"

    except Exception as e:
        return f"[ERROR READING LINK]: {str(e)}"

# --- HELPER: PARSE MARKDOWN TABLES ---
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

# --- FILE FACTORY ---
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
                for i, row_data in enumerate(table_data):
                    row = table.rows[i]
                    for j, text in enumerate(row_data): row.cells[j].text = str(text)
                for line in lines:
                    if not "|" in line and not "---" in line and line.strip():
                        if line.startswith("#"): doc.add_heading(line.replace("#", "").strip(), level=1)
                        else: doc.add_paragraph(line)
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
                    for row_data in table_data:
                        r = table.row()
                        for item in row_data: r.cell(str(item))
                pdf.ln(10)
                for line in lines:
                    if not "|" in line and not "---" in line and line.strip():
                        safe_text = line.encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 7, safe_text)
            else:
                for line in lines:
                    if line.strip():
                        safe_text = line.encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 7, safe_text)
            return base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('utf-8'), f"{filename}.pdf"
            
        elif file_type == "txt":
            return base64.b64encode(content.encode('utf-8')).decode('utf-8'), f"{filename}.txt"
    except Exception as e:
        logger.critical(f"File Gen Error: {e}")
        return base64.b64encode(f"Error: {content}".encode('utf-8')).decode('utf-8'), f"{filename}_fallback.txt"
    return None, None

def generate_high_quality_image(prompt):
    # (Same as previous, omitted for brevity but assumed present in final file)
    if not HUGGINGFACE_API_KEY: return None, None
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    try:
        # Simplified check
        res = requests.post("https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev", headers=headers, json={"inputs": prompt})
        if res.status_code == 200: 
            return base64.b64encode(res.content).decode('utf-8'), "gen.jpg"
    except: pass
    return None, None

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

def generate_voice(text):
    if not ELEVENLABS_API_KEY: return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{BRIAN_VOICE_ID}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    try:
        res = requests.post(url, json={"text": text[:1000], "model_id": "eleven_monolingual_v1"}, headers=headers)
        if res.status_code == 200: return base64.b64encode(res.content).decode('utf-8')
    except: pass
    return None

@app.get("/")
def home():
    return {"status": "Alfred V3.0 Online (Real-Time Intelligence)"}

@app.post("/command")
def process_command(request: UserRequest, x_alfred_auth: Optional[str] = Header(None)):
    if x_alfred_auth != SERVER_SECRET_KEY:
        raise HTTPException(status_code=401, detail="ACCESS DENIED")

    if not client: return {"response": "Sir, connection severed (No API Key)."}

    try:
        chat_history = []
        for msg in request.history:
            role = "model" if msg.role == "alfred" else "user"
            chat_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        # --- 1. REAL-TIME DATA AGGREGATION ---
        system_context = []
        
        # Check for Link
        scraped_content = get_url_content(request.command)
        if scraped_content: system_context.append(scraped_content)
        
        # Check for Stocks
        stock_data = get_stock_data(request.command)
        if stock_data: system_context.append(stock_data)
        
        # Check for News
        news_data = get_news_feed(request.command)
        if news_data: system_context.append(news_data)
        
        # Check for Weather
        weather_data = get_weather(request.command)
        if weather_data: system_context.append(weather_data)
        
        # Check for Sentiment request
        sentiment_data = analyze_sentiment(request.command)
        if sentiment_data: system_context.append(sentiment_data)

        # Construct Final Prompt
        sa_timezone = pytz.timezone('Africa/Johannesburg')
        now = datetime.now(sa_timezone).strftime("%A, %B %d, %Y at %I:%M %p (SAST)")
        
        final_prompt = f"[Current Time: {now}]\n"
        if system_context:
            final_prompt += "[REAL-TIME DATA INJECTED]:\n" + "\n".join(system_context) + "\n\n"
        
        final_prompt += f"User Query: {request.command}"

        current_parts = [final_prompt]
        if request.file_data:
            try:
                file_bytes = base64.b64decode(request.file_data)
                current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=request.mime_type))
            except: pass

        # Thinking model is best for Synthesis tasks
        selected_model = 'gemini-2.0-flash-thinking-exp-01-21' if request.thinking_mode else 'gemini-2.0-flash'

        # Retry Loop
        response = None
        for attempt in range(3):
            try:
                chat_session = client.chats.create(
                    model=selected_model,
                    history=chat_history,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        system_instruction=ALFRED_SYSTEM_INSTRUCTIONS
                    )
                )
                response = chat_session.send_message(message=current_parts)
                break
            except Exception as e:
                if attempt == 2: raise e
                time.sleep(1)
        
        reply = response.text
        gen_file_data = None
        gen_filename = None

        # File/Image/Chart Handlers (Standard)
        if "<<<FILE_START>>>" in reply:
            try:
                content = reply.split("<<<FILE_START>>>")[1].split("<<<FILE_END>>>")[0].strip()
                requested_type = "txt"
                if "pdf" in request.command.lower(): requested_type = "pdf"
                elif "word" in request.command.lower(): requested_type = "docx"
                
                gen_file_data, gen_filename = create_file(content, requested_type)
                reply = f"I have synthesized the data into a {requested_type.upper()} file for you, Sir."
            except: reply = "File generation failed."

        elif "IMAGE_GEN_REQUEST:" in reply:
            image_prompt = reply.split("IMAGE_GEN_REQUEST:")[1].strip()
            gen_file_data, gen_filename = generate_high_quality_image(image_prompt)
            reply = "Visualizing concept..."

        elif "<<<CHART_START>>>" in reply:
            try:
                code = reply.split("<<<CHART_START>>>")[1].split("<<<CHART_END>>>")[0].strip()
                gen_file_data, gen_filename = execute_chart_code(code)
                reply = "Data visualization generated."
            except: pass

        audio_data = generate_voice(reply) if request.speak else None

    except Exception as e:
        logger.error(f"Error: {e}")
        reply = f"SYSTEM ERROR: {str(e)}"
        audio_data, gen_file_data, gen_filename = None, None, None

    return {"response": reply, "audio": audio_data, "file": {"data": gen_file_data, "name": gen_filename} if gen_file_data else None}
