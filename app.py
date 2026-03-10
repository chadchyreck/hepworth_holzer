import os
import re
import glob
import openai
import requests
import pandas as pd
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from docx import Document as DocxDocument

# ========== SETUP ==========
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'replace-this-with-something-very-secret')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

openai.api_key = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# ========== FIRM CONSTANTS ==========
FIRM_NAME = "Hepworth Holzer"
FIRM_PHONE = "208-328-6998"
FIRM_WEBSITE = "https://hepworthholzer.com/"
FIRM_ADDRESS_BOISE = "537 W. Bannock Street #200, Boise, Idaho 83702"
FIRM_ADDRESS_MERIDIAN = "1910 N. Lakes Pl. Suite A, Meridian, Idaho 83646"

DISCLAIMER = (
    "⚖️ *Disclaimer: This does not constitute legal advice. "
    "Always consult with a qualified trial lawyer in your area about your specific circumstance.*"
)

WELCOME_MESSAGE = (
    f"{DISCLAIMER}\n\n"
    f"Hello! Welcome to the **{FIRM_NAME} AI Legal Assistant**. 👋\n\n"
    "I'm here to help answer your questions about Idaho personal injury law and point you "
    "in the right direction. Whether you've been injured in a car accident, a slip and fall, "
    "or another incident — I'm here to help.\n\n"
    "Please tell me what happened or ask me a question to get started."
)

MED_MAL_KEYWORDS = [
    "medical malpractice", "medical negligence", "doctor mistake", "surgical error",
    "misdiagnosis", "hospital negligence", "nursing home abuse", "birth injury",
    "wrong medication", "anesthesia error", "medical error"
]

OUT_OF_SCOPE_KEYWORDS = [
    "criminal", "divorce", "custody", "dui", "dwi", "bankruptcy", "immigration",
    "family law", "contract", "employment", "workers comp", "out of state",
    "washington", "oregon", "nevada", "utah", "wyoming", "montana"
]

# ========== LOAD KNOWLEDGE BASE ==========

def clean_text(text):
    """Strip HTML tags, shortcodes, and extra whitespace."""
    text = re.sub(r'\[.*?\]', '', text)           # remove shortcodes
    text = re.sub(r'<[^>]+>', '', text)            # remove HTML tags
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)       # remove HTML entities
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_word_docs(folder="knowledge_base"):
    """Load all .docx files from the knowledge_base folder."""
    docs = {}
    pattern = os.path.join(folder, "*.docx")
    for filepath in glob.glob(pattern):
        try:
            doc = DocxDocument(filepath)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            filename = os.path.basename(filepath)
            docs[filename] = clean_text(text)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return docs

def load_blog_excel(filepath="knowledge_base/blogs.csv"):
    """Load blog posts from CSV file with columns: ID, Title, Content."""
    blogs = []
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            title = str(row.get("Title", "")).strip()
            content = clean_text(str(row.get("Content", "")))
            if title and content:
                blogs.append({"title": title, "content": content})
        print(f"Loaded {len(blogs)} blog posts.")
    except Exception as e:
        print(f"Error loading blog Excel: {e}")
    return blogs

# Load on startup
WORD_DOCS = load_word_docs()
BLOG_POSTS = load_blog_excel()

print(f"Knowledge base loaded: {len(WORD_DOCS)} Word docs, {len(BLOG_POSTS)} blog posts.")

# ========== KNOWLEDGE BASE SEARCH ==========

def search_knowledge_base(query, top_n=5):
    """Simple keyword search across Word docs and blog posts."""
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    results = []

    # Search Word docs
    for filename, content in WORD_DOCS.items():
        content_lower = content.lower()
        matches = sum(1 for w in query_words if w in content_lower)
        if matches > 0:
            results.append({
                "source": filename,
                "content": content[:2000],
                "score": matches
            })

    # Search blog posts
    for blog in BLOG_POSTS:
        combined = (blog["title"] + " " + blog["content"]).lower()
        matches = sum(1 for w in query_words if w in combined)
        if matches > 0:
            results.append({
                "source": blog["title"],
                "content": blog["content"][:1500],
                "score": matches
            })

    # Sort by relevance score, return top N
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]

def get_med_mal_notice():
    """Retrieve the Medical Negligence Notice doc content."""
    for filename, content in WORD_DOCS.items():
        if "medical negligence notice" in filename.lower():
            return content
    return None

# ========== SERPER SEARCH ==========

def search_serper(query):
    """Search broadly for Idaho personal injury law topics."""
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    search_query = f"{query} Idaho personal injury law"
    payload = {"q": search_query}
    try:
        resp = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
        if resp.status_code != 200:
            return ""
        results = resp.json().get("organic", [])
        snippets = [
            f"- {r['title']}: {r.get('snippet', '')}"
            for r in results[:3]
        ]
        return "\n".join(snippets)
    except Exception as e:
        print(f"Serper error: {e}")
        return ""

def remove_urls(text):
    """Remove any raw URLs that slip into responses."""
    return re.sub(r'http[s]?://\S+|www\.\S+', '', text)

# ========== DETECTION HELPERS ==========

def is_med_mal_query(message):
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in MED_MAL_KEYWORDS)

def is_out_of_scope(message):
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in OUT_OF_SCOPE_KEYWORDS)

# ========== SYSTEM PROMPT ==========

def build_system_prompt(kb_context, serper_context, is_med_mal, med_mal_summary):
    base = f"""You are the {FIRM_NAME} AI Legal Assistant — a warm, empathetic, and knowledgeable guide for people who have been injured in Idaho.

FIRM INFORMATION:
- Name: {FIRM_NAME}
- Phone: {FIRM_PHONE}
- Website: {FIRM_WEBSITE}
- Offices: {FIRM_ADDRESS_BOISE} and {FIRM_ADDRESS_MERIDIAN}
- Practice: Idaho personal injury law only

YOUR ROLE:
- Answer questions about Idaho personal injury law with warmth and empathy
- Your PRIMARY job is to EDUCATE and INFORM the user using the knowledge base content and blog posts provided
- Always lead with substantive, helpful legal information drawn from the knowledge base BEFORE mentioning contacting the firm
- Give thorough, detailed answers about Idaho personal injury law — explain the process, what to expect, what factors matter, timelines, and what steps someone should take
- Only suggest contacting the firm AFTER you have fully answered the question with real information
- Never cut an answer short just to push someone to call — that is unhelpful and will frustrate users
- Always acknowledge the emotional difficulty of being injured before diving into information
- For out-of-scope questions (criminal law, divorce, cases outside Idaho, etc.), politely decline and suggest they find qualified local counsel

RESPONSE STRUCTURE — follow this order every time:
1. Acknowledge their situation with empathy (1-2 sentences)
2. Provide detailed, substantive information from the knowledge base about their topic (this should be the bulk of your response)
3. Explain what factors are important, what the process looks like, what they should know
4. Only at the END, after fully answering, mention that Hepworth Holzer is available if they want to discuss their specific situation — include phone ({FIRM_PHONE}) and website ({FIRM_WEBSITE}) for non-medical-malpractice cases

MEDICAL MALPRACTICE — special handling:
- Hepworth Holzer handles very limited medical malpractice cases due to Idaho law's strict requirements
- When someone asks about medical malpractice or medical negligence, FIRST provide the full detailed information from the Medical Negligence Notice — explain all the key requirements, deadlines, and complexities in plain English
- Make clear what makes these cases difficult in Idaho and what criteria must be met
- Only AFTER fully explaining all of this, mention they can call to discuss whether their situation might qualify
- Do NOT provide the website for medical malpractice — only the phone number
- Never give false hope about medical malpractice cases

TONE:
- Warm, empathetic, and human — acknowledge pain and difficulty first
- Informative and thorough — users came here for real answers, give them real answers
- Never make the user feel like they are being pushed to call before getting help

IMPORTANT:
- Never give specific legal advice or predict case outcomes
- Do not fabricate laws, statutes, or case facts
- Do not include raw URLs in your response text
- Keep responses well-organized and digestible — use short paragraphs, not walls of text
- The contact information should always be the LAST thing in your response, never the first
"""

    if kb_context:
        base += f"\n\nKNOWLEDGE BASE CONTEXT (use this to inform your answer):\n{kb_context}"

    if serper_context:
        base += f"\n\nWEB SEARCH CONTEXT (use as supplemental reference):\n{serper_context}"

    if is_med_mal and med_mal_summary:
        base += f"\n\nMEDICAL NEGLIGENCE NOTICE SUMMARY FOR THIS RESPONSE:\n{med_mal_summary}"

    return base

# ========== WELCOME ENDPOINT ==========

@app.route('/welcome', methods=['GET'])
def welcome():
    return jsonify({"message": WELCOME_MESSAGE})

# ========== CHAT ENDPOINT ==========

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"reply": "I didn't catch that — could you tell me a little more about your situation?"})

    # --- Out of scope check ---
    if is_out_of_scope(user_message):
        reply = (
            "That's a great question, but it falls outside the areas I can help with. "
            f"{FIRM_NAME} focuses exclusively on personal injury law in Idaho. "
            "For other legal matters, I'd encourage you to reach out to a qualified attorney "
            "in that specific area of law in your region. If you have any Idaho personal injury "
            "questions, I'm happy to help with those!"
        )
        return jsonify({"reply": reply})

    # --- Medical malpractice check ---
    med_mal = is_med_mal_query(user_message)
    med_mal_summary = ""
    if med_mal:
        notice_content = get_med_mal_notice()
        if notice_content:
            # Ask GPT to summarize the notice key points
            try:
                summary_resp = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a legal assistant. Summarize the key points of the following Medical Negligence Notice in plain English, in 4-6 concise bullet points. Do not include URLs."},
                        {"role": "user", "content": notice_content[:4000]}
                    ],
                    max_tokens=400,
                    temperature=0.2
                )
                med_mal_summary = summary_resp.choices[0].message.content
            except Exception as e:
                med_mal_summary = notice_content[:1000]

    # --- Knowledge base search ---
    kb_results = search_knowledge_base(user_message)
    kb_context = ""
    if kb_results:
        kb_context = "\n\n---\n\n".join(
            [f"Source: {r['source']}\n{r['content']}" for r in kb_results]
        )

    # --- Serper fallback (always run as supplement) ---
    serper_context = search_serper(user_message)

    # --- Build system prompt ---
    system_prompt = build_system_prompt(kb_context, serper_context, med_mal, med_mal_summary)

    # --- Build message history ---
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})
    messages.append({"role": "user", "content": user_message})

    # --- Call OpenAI ---
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1200,
            temperature=0.35
        )
        reply = completion.choices[0].message.content
        reply = remove_urls(reply)
    except Exception as e:
        reply = f"I'm sorry, I encountered an error. Please call us directly at {FIRM_PHONE} for immediate assistance."

    return jsonify({"reply": reply})

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)