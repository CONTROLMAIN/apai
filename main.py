import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from supabase import create_client, Client

app = FastAPI()

# ===== ENV VARS (set these on Railway later) =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
INGEST_SECRET = os.getenv("INGEST_SECRET", "change-me")

if not (OPENAI_API_KEY and SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
    # Don't crash locally if you're still setting up, but /ingest and /ask will fail.
    pass

def sb() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

SUBJECT_FOLDERS = {
    "ap_precalc": "data/ap_precalc",
    "ap_physics": "data/ap_physics",
}

class AskBody(BaseModel):
    subject: str  # "ap_precalc" or "ap_physics"
    question: str

class IngestBody(BaseModel):
    subject: str  # "ap_precalc" or "ap_physics"

@app.get("/health")
def health():
    return {"ok": True}

def pdf_to_text_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages

@app.post("/ingest")
def ingest(body: IngestBody, x_ingest_secret: str = Header(default="")):
    if x_ingest_secret != INGEST_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    subject = body.subject
    if subject not in SUBJECT_FOLDERS:
        raise HTTPException(status_code=400, detail="Unknown subject")

    folder = SUBJECT_FOLDERS[subject]
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Missing folder: {folder}")

    # Collect all text
    docs = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        for page_no, text in pdf_to_text_pages(path):
            if text.strip():
                docs.append({"subject": subject, "source_file": fname, "page": page_no, "text": text})

    if not docs:
        raise HTTPException(status_code=400, detail="No text extracted. Are PDFs scanned?")

    # Chunk
    chunks = []
    for d in docs:
        split = splitter.split_text(d["text"])
        for idx, ch in enumerate(split):
            chunks.append({
                "subject": d["subject"],
                "source_file": d["source_file"],
                "page": d["page"],
                "chunk_index": idx,
                "content": ch
            })

    # Embed + store in Supabase
    # Expect a table like: documents(id, content, embedding, metadata jsonb)
    # Your Supabase LangChain quickstart provides the schema / helper function.
    supa = sb()

    # Optional: delete old subject rows to avoid duplicates
    supa.table("documents").delete().eq("metadata->>subject", subject).execute()

    # Insert in batches
    BATCH = 50
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        texts = [b["content"] for b in batch]
        vecs = embeddings.embed_documents(texts)

        rows = []
        for b, v in zip(batch, vecs):
            rows.append({
                "content": b["content"],
                "embedding": v,
                "metadata": {
                    "subject": b["subject"],
                    "source_file": b["source_file"],
                    "page": b["page"],
                    "chunk_index": b["chunk_index"],
                }
            })
        supa.table("documents").insert(rows).execute()

    return {"ingested_chunks": len(chunks), "subject": subject}

def retrieve(subject: str, query: str, k: int = 6):
    # Supabase quickstart typically provides a match function like match_documents(query_embedding, match_count, filter)
    # We will call a function named match_documents (common pattern).
    qvec = embeddings.embed_query(query)
    supa = sb()
    resp = supa.rpc("match_documents", {
        "query_embedding": qvec,
        "match_count": k,
        "filter": {"subject": subject}
    }).execute()

    data = resp.data or []
    return data

@app.post("/ask")
def ask(body: AskBody):
    subject = body.subject
    question = body.question.strip()
    if subject not in SUBJECT_FOLDERS:
        raise HTTPException(status_code=400, detail="Unknown subject")
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    hits = retrieve(subject, question, k=6)
    if not hits:
        return {"answer": "Not found in the provided AP sources.", "sources": []}

    context_blocks = []
    sources = []
    for i, h in enumerate(hits, start=1):
        md = h.get("metadata", {})
        sources.append(md)
        context_blocks.append(
            f"[{i}] {md.get('source_file','?')} p.{md.get('page','?')}\n{h.get('content','')}"
        )

    system = f"""
You are APAI, an AP-aligned assistant for {subject}.
Rules:
- Use ONLY the SOURCES below. If not supported, say: "Not found in the provided AP sources."
- Be AP-level and step-by-step.
- End with a short "Sources used: [..]" list matching the source numbers.
"""

    prompt = system + "\n\nQUESTION:\n" + question + "\n\nSOURCES:\n" + "\n\n".join(context_blocks)
    resp = llm.invoke(prompt)

    return {"answer": resp.content, "sources": sources}
