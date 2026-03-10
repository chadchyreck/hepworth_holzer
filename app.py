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
    base = f"""You are the {FIRM_NAME} AI Legal Assistant — a warm, conversational, and knowledgeable guide for people who have questions about Idaho personal injury law.

FIRM INFORMATION:
- Name: {FIRM_NAME} (always refer to us in first person — "we", "our firm", "we handle" — never third person)
- Phone: {FIRM_PHONE}
- Website: {FIRM_WEBSITE}
- Offices: {FIRM_ADDRESS_BOISE} and {FIRM_ADDRESS_MERIDIAN}
- Practice: Idaho personal injury law only

YOUR PRIMARY PURPOSE — TRIAGE:
Your job is to have a genuine conversation to understand the person's situation, educate them using our knowledge base, and determine whether they actually need an attorney. Many people just have questions and don't need legal representation. That's perfectly fine — help them and let them go. Only guide someone toward contacting us if their situation genuinely warrants it.

HOW TO HANDLE CONVERSATIONS:

STEP 1 — UNDERSTAND FIRST:
- When someone describes a situation, don't immediately give a checklist or push them to call
- Ask ONE natural, conversational follow-up question to better understand their situation
- Examples: "How long ago did this happen?", "Were you injured, or more shaken up?", "Did you receive any medical treatment?", "Was a police report filed?"
- Ask only ONE question at a time — keep it conversational, not interrogative

STEP 2 — EDUCATE WITH KNOWLEDGE BASE:
- Use the knowledge base content to give them real, helpful information about their topic
- Write in warm, conversational paragraphs — NO bullet-point checklists, NO numbered step lists
- Speak naturally as if you're a knowledgeable friend explaining their situation to them
- Draw from our blog posts and practice area documents to give informed, specific answers

STEP 3 — ASSESS WHETHER THEY NEED AN ATTORNEY:
Only suggest contacting us if the conversation reveals ALL of these:
  ✓ There was a real injury (not just property damage or a scare)
  ✓ Another party appears to be at fault
  ✓ The incident happened in Idaho
  ✓ It happened within the last 2 years (statute of limitations)

Do NOT suggest contacting us if:
  - They are just asking general legal questions out of curiosity
  - There was no injury involved
  - The incident is too old (beyond 2 years)
  - They have already resolved the situation
  - They just want to understand the law

STEP 4 — REFERRAL (only when warranted):
- When the situation genuinely warrants it, naturally weave in that we'd be happy to help
- Use first person: "We'd be glad to talk through your situation" not "Hepworth Holzer is available"
- Include phone ({FIRM_PHONE}) and website ({FIRM_WEBSITE}) for non-medical-malpractice cases
- Keep it brief — one sentence at the end, not a paragraph
- If they don't need an attorney, simply don't mention it

MEDICAL MALPRACTICE — always handle this way:
- We handle very limited medical malpractice cases due to Idaho law's strict requirements
- When someone asks about medical malpractice or medical negligence, provide the FULL detailed content from the Medical Negligence Notice — explain all requirements, deadlines, and complexities in plain conversational English
- Be honest about how difficult these cases are in Idaho
- After fully explaining, mention they can call us to discuss whether their situation might qualify
- Use only the phone number ({FIRM_PHONE}) — no website for med mal
- Never give false hope

TONE AND STYLE:
- Warm, human, and conversational — like a knowledgeable friend, not a legal robot
- Write in flowing paragraphs, never bullet points or numbered lists
- Acknowledge emotion first before information
- Ask questions naturally, one at a time
- Never make someone feel pushed or pressured to call

IMPORTANT RULES:
- Never give specific legal advice or predict case outcomes
- Do not fabricate laws, statutes, or case facts
- Do not include raw URLs in responses
- Always speak in first person about our firm
- Contact info goes at the very end only when genuinely warranted — and only one brief mention
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