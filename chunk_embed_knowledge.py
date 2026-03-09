import os
import json
import openai
import pandas as pd
import glob
import re
import nltk
from docx import Document as DocxDocument
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# === CONFIG ===
KNOWLEDGE_FOLDER = "knowledge_base"          # Folder containing .docx files and blogs CSV/Excel
BLOG_FILE_CSV   = os.path.join(KNOWLEDGE_FOLDER, "blogs.csv")    # CSV export (primary)
BLOG_FILE_XLSX  = os.path.join(KNOWLEDGE_FOLDER, "blogs.xlsx")   # Excel fallback
OUTPUT_FILE     = "doc_embeddings.json"
CHUNK_SIZE      = 400    # Words per chunk — good balance for legal Q&A retrieval
MIN_CHUNK_SIZE  = 70     # Discard chunks shorter than this
CHUNK_OVERLAP   = 80     # Word overlap between chunks to preserve context across boundaries

openai.api_key = os.getenv("OPENAI_API_KEY")

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_legal_text(text):
    """
    Clean and normalize text extracted from legal documents and blog posts.

    Handles everything found in the Hepworth Holzer blog CSV export:
      - <style> and <script> blocks
      - MS Office XML conditional comments  <!--[if gte mso 9]>...</[endif]-->
      - Office XML namespace tags  <o:...>  <w:...>  <m:...>
      - All HTML tags and entities  (&amp; &#160; etc.)
      - WordPress block shortcodes  [block id="..."]  [ux_video url="..."]
      - Layout shortcodes  [row]  [/row]  [col]  [/col]  [section]  [caption]
      - Raw URLs and email addresses
      - Non-breaking spaces (\xa0) and Windows line endings (\r\n)
    """
    if not text:
        return ""

    # --- 1. Block-level markup (must come before generic tag stripping) ---

    # Remove <style>...</style> blocks entirely
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove <script>...</script> blocks entirely
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove MS Office XML conditional comments <!--[if gte mso 9]>...<![endif]-->
    text = re.sub(r'<!--\[if[^\]]*\]>.*?<!\[endif\]-->', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove Office XML namespace element pairs <o:Tag>...</o:Tag> / <w:Tag> / <m:Tag>
    text = re.sub(r'<[owm]:[^>]+>.*?</[owm]:[^>]+>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove self-closing Office XML tags <o:Tag/> <w:Tag attr="x"/>
    text = re.sub(r'<[owm]:[^>]+/?>', ' ', text, flags=re.IGNORECASE)

    # --- 2. All remaining HTML tags ---
    text = re.sub(r'<[^>]+>', ' ', text)

    # --- 3. HTML entities ---
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)

    # --- 4. WordPress / page-builder shortcodes ---

    # Named shortcodes with attributes (opening and self-closing):
    #   [block id="blog-attorneys"]  [ux_video url="..."]  [caption id="..." ...]
    #   [row width="full-width"]  [col span="12"]  [section padding="0px"]
    text = re.sub(
        r'\[/?(?:block|ux_video|row|col|section|caption|gallery|embed|video|audio|'
        r'contact|form|button|icon|wp_\w+)[^\]]*\]',
        ' ', text, flags=re.IGNORECASE
    )

    # Any remaining closing shortcodes [/anything]
    text = re.sub(r'\[/\w+\]', ' ', text)

    # --- 5. URLs and emails ---
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # --- 6. Special whitespace characters ---
    text = text.replace('\xa0', ' ')   # non-breaking space
    text = text.replace('\r\n', ' ')   # Windows line endings
    text = text.replace('\r', ' ')

    # --- 7. Typography clean-up ---
    # Fix run-together words caused by stripped tags (e.g. "wordWord")
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)

    # Capitalise first letter after sentence-ending punctuation
    text = re.sub(
        r'([.!?])\s*([a-z])',
        lambda m: m.group(1) + ' ' + m.group(2).upper(),
        text
    )

    # Normalise bullet and list markers
    text = re.sub(r'^\s*[•\-\*]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)

    # Collapse multiple periods and spaces
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)

    # --- 8. Final trim and punctuation ---
    text = text.strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'

    return text


# ============================================================
# EXTRACTION — WORD DOCUMENTS
# ============================================================

def extract_docx(file_path):
    """Extract and clean text from a .docx file."""
    try:
        doc = DocxDocument(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        raw = "\n\n".join(paragraphs)
        return clean_legal_text(raw)
    except Exception as e:
        print(f"  ⚠️  DOCX extraction error for {file_path}: {e}")
        return ""


# ============================================================
# EXTRACTION — BLOG POSTS (CSV or XLSX)
# ============================================================

def load_blog_posts():
    """
    Load blog posts from CSV (preferred) or Excel fallback.
    Expects columns: id (ignored), Title, Content.
    Returns a list of dicts: {title, content, source}.
    """
    df = None

    # Try CSV first
    if os.path.exists(BLOG_FILE_CSV):
        try:
            df = pd.read_csv(BLOG_FILE_CSV)
            print(f"  📄 Loaded blog CSV: {BLOG_FILE_CSV}")
        except Exception as e:
            print(f"  ⚠️  CSV load error: {e}")

    # Fall back to Excel
    if df is None and os.path.exists(BLOG_FILE_XLSX):
        try:
            df = pd.read_excel(BLOG_FILE_XLSX)
            print(f"  📄 Loaded blog Excel: {BLOG_FILE_XLSX}")
        except Exception as e:
            print(f"  ⚠️  Excel load error: {e}")

    if df is None:
        print(f"  ⚠️  No blog file found at '{BLOG_FILE_CSV}' or '{BLOG_FILE_XLSX}'. Skipping blogs.")
        return []

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    posts = []
    for _, row in df.iterrows():
        title   = str(row.get("Title",   "")).strip()
        content = clean_legal_text(str(row.get("Content", "")))
        # Skip column 'id' entirely as instructed
        if title and content and len(content) > MIN_CHUNK_SIZE:
            posts.append({
                "title":   title,
                "content": content,
                "source":  f"blog: {title}"
            })

    print(f"  ✅ {len(posts)} usable blog posts loaded.")
    return posts


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, source_label=""):
    """
    Split text into overlapping sentence-aware chunks.
    Overlap preserves legal context across chunk boundaries.
    """
    if not text or len(text.strip()) < MIN_CHUNK_SIZE:
        return []

    # Sentence tokenise
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)

    sentences = [s.strip() for s in sentences if s.strip()]

    chunks        = []
    current       = []
    current_words = 0

    for sentence in sentences:
        sw = len(sentence.split())

        if current_words + sw > chunk_size and current:
            chunk_text_str = " ".join(current)
            if len(chunk_text_str.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(chunk_text_str.strip())

            # Roll back by `overlap` words to maintain context
            overlap_sents  = []
            overlap_words  = 0
            for s in reversed(current):
                wc = len(s.split())
                if overlap_words + wc <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_words += wc
                else:
                    break

            current       = overlap_sents
            current_words = overlap_words

        current.append(sentence)
        current_words += sw

    # Final chunk
    if current:
        chunk_text_str = " ".join(current)
        if len(chunk_text_str.strip()) >= MIN_CHUNK_SIZE:
            chunks.append(chunk_text_str.strip())

    return chunks


# ============================================================
# EMBEDDING
# ============================================================

def embed_chunks(chunks, source_label):
    """
    Call the OpenAI embeddings API for each chunk.
    Handles rate limits with a simple retry.
    """
    embedded = []

    for i, chunk in enumerate(tqdm(chunks, desc=f"  Embedding: {source_label[:55]}")):
        def _call():
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            return response.data[0].embedding

        try:
            embedding = _call()
        except openai.RateLimitError:
            print("  ⏳ Rate limit hit — waiting 20 s...")
            import time; time.sleep(20)
            try:
                embedding = _call()
            except Exception as e:
                print(f"  ❌ Retry failed for chunk {i+1} of '{source_label}': {e}")
                continue
        except Exception as e:
            print(f"  ❌ Embedding failed for chunk {i+1} of '{source_label}': {e}")
            continue

        embedded.append({
            "text":       chunk,
            "embedding":  embedding,
            "source":     source_label,
            "chunk_id":   f"{source_label}_chunk_{i+1}",
            "word_count": len(chunk.split()),
            "char_count": len(chunk)
        })

    return embedded


# ============================================================
# PREVIEW
# ============================================================

def preview_chunks(chunks, label, max_preview=2):
    print(f"\n  📝 Preview — {label}:")
    for i, chunk in enumerate(chunks[:max_preview]):
        preview = chunk[:220] + "..." if len(chunk) > 220 else chunk
        print(f"    Chunk {i+1} ({len(chunk.split())} words): {preview}")
    if len(chunks) > max_preview:
        print(f"    ... and {len(chunks) - max_preview} more chunks")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("⚖️  Starting Hepworth Holzer AI Legal Assistant — knowledge base processing...\n")

    if not os.path.exists(KNOWLEDGE_FOLDER):
        print(f"❌ Knowledge folder '{KNOWLEDGE_FOLDER}' not found!")
        print("Please create the folder and add the following files:")
        print("  • Personal_Injury.docx")
        print("  • Bicycle_Accidents.docx")
        print("  • Car_Accidents.docx")
        print("  • Catastrophic_Injuries.docx")
        print("  • Dog_Bites.docx")
        print("  • Drunk_Driving_Injuries.docx")
        print("  • Medical_Negligence_Notice.docx")
        print("  • Motorcycle_Accidents.docx")
        print("  • Pedestrian_Accidents.docx")
        print("  • Product_Liability.docx")
        print("  • Slip_and_Fall.docx")
        print("  • Truck_Accident.docx")
        print("  • Uninsured_Motorist.docx")
        print("  • Wrongful_Death.docx")
        print("  • blogs.csv  (or blogs.xlsx) — columns: id, Title, Content")
        exit(1)

    all_embeddings = []

    # ----------------------------------------------------------
    # 1. PROCESS WORD DOCUMENTS
    # ----------------------------------------------------------
    print("=" * 60)
    print("📁 Processing Word documents...")
    print("=" * 60)

    docx_files = sorted(glob.glob(os.path.join(KNOWLEDGE_FOLDER, "*.docx")))

    if not docx_files:
        print(f"  ⚠️  No .docx files found in '{KNOWLEDGE_FOLDER}'")
    else:
        print(f"  Found {len(docx_files)} Word document(s)\n")

    for filepath in docx_files:
        filename = os.path.basename(filepath)
        print(f"\n📄 {filename}")

        text = extract_docx(filepath)
        if not text.strip():
            print(f"  ⚠️  Skipped — empty or unreadable file.")
            continue

        print(f"  ✅ Extracted {len(text):,} characters")

        chunks = chunk_text(text, source_label=filename)
        if not chunks:
            print(f"  ⚠️  No valid chunks produced.")
            continue

        print(f"  📦 {len(chunks)} chunks created")
        preview_chunks(chunks, filename)

        embeddings = embed_chunks(chunks, source_label=filename)
        all_embeddings.extend(embeddings)
        print(f"  ✅ {len(embeddings)} embeddings stored")

    # ----------------------------------------------------------
    # 2. PROCESS BLOG POSTS
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("📰 Processing blog posts...")
    print("=" * 60)

    blog_posts = load_blog_posts()

    for post in tqdm(blog_posts, desc="  Processing blogs"):
        # For short posts, treat the whole post as one chunk
        # For longer posts, chunk normally
        combined_text = f"{post['title']}. {post['content']}"
        chunks = chunk_text(combined_text, source_label=post["source"])

        if not chunks:
            continue

        embeddings = embed_chunks(chunks, source_label=post["source"])
        all_embeddings.extend(embeddings)

    # ----------------------------------------------------------
    # 3. SAVE OUTPUT
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"💾 Saving embeddings to '{OUTPUT_FILE}'...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_embeddings, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------
    # 4. SUMMARY
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"🎉  HEPWORTH HOLZER KNOWLEDGE BASE COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  📦 Total embeddings : {len(all_embeddings):,}")
    print(f"  💾 Output file      : {OUTPUT_FILE}")

    if all_embeddings:
        word_counts = [item["word_count"] for item in all_embeddings]
        avg_words   = sum(word_counts) / len(word_counts)
        print(f"  📈 Avg chunk size   : {avg_words:.1f} words")
        print(f"  📏 Chunk range      : {min(word_counts)} – {max(word_counts)} words")

    print(f"\n⚖️  The Hepworth Holzer AI Legal Assistant knowledge base is ready.")
    print(f"    Covers: personal injury, car/truck/motorcycle/bicycle/pedestrian accidents,")
    print(f"    dog bites, drunk driving injuries, slip & fall, product liability,")
    print(f"    catastrophic injuries, wrongful death, uninsured motorist,")
    print(f"    medical negligence, and {len(blog_posts):,} blog posts.")
