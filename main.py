import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from pydantic import BaseModel
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()

BASE = Path(__file__).parent
PDF_DIR = BASE / "pdfs"
STATIC_DIR = BASE / "static"
MODEL = os.getenv("MODEL", "gpt-4o-mini")

CHUNK_WORDS = 220  # words per chunk
OVERLAP = 40       # words shared between neighbouring chunks
TOP_K = 6          # chunks sent to Claude per question

SYSTEM_PROMPT = (
    "You answer questions using only the document excerpts provided. "
    "Each excerpt is labelled with its file name and page. "
    "Cite the sources you used in the form [file.pdf, p.3]. "
    "If the excerpts do not contain the answer, say so plainly and suggest "
    "what the person could ask instead. Keep answers clear and concise."
)

client = OpenAI()  # reads OPENAI_API_KEY from the environment


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_chunks() -> list[dict]:
    chunks = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf))
        except Exception as exc:
            print(f"Skipping {pdf.name}: {exc}")
            continue
        for page_no, page in enumerate(reader.pages, start=1):
            words = (page.extract_text() or "").split()
            step = CHUNK_WORDS - OVERLAP
            for i in range(0, len(words), step):
                piece = " ".join(words[i : i + CHUNK_WORDS])
                if len(piece) > 50:
                    chunks.append({"source": pdf.name, "page": page_no, "text": piece})
    return chunks


CHUNKS = load_chunks()
INDEX = BM25Okapi([tokenize(c["text"]) for c in CHUNKS]) if CHUNKS else None
print(f"Indexed {len(CHUNKS)} chunks from {len(list(PDF_DIR.glob('*.pdf')))} PDFs")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


app = FastAPI(title="PDF Bot")


@app.get("/api/health")
def health():
    return {"ok": True, "chunks": len(CHUNKS), "model": MODEL}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not CHUNKS or INDEX is None:
        raise HTTPException(503, "No PDFs indexed. Add PDF files to the pdfs/ folder and restart.")

    history = [m for m in req.messages if m.role in ("user", "assistant") and m.content.strip()]
    history = history[-10:]
    if not history or history[-1].role != "user":
        raise HTTPException(400, "The last message must be from the user.")

    # Search with the last two user messages so follow-up questions still find context.
    user_turns = [m.content for m in history if m.role == "user"]
    query = " ".join(user_turns[-2:])
    scores = INDEX.get_scores(tokenize(query))
    top = sorted(range(len(CHUNKS)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    top = [i for i in top if scores[i] > 0]

    context = "\n\n".join(
        f"[{CHUNKS[i]['source']}, p.{CHUNKS[i]['page']}]\n{CHUNKS[i]['text']}" for i in top
    ) or "(no matching excerpts found)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in history[:-1]]
    messages.append(
        {
            "role": "user",
            "content": f"Document excerpts:\n\n{context}\n\nQuestion: {history[-1].content}",
        }
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=messages,
        )
    except OpenAIError as exc:
        raise HTTPException(502, f"The model request failed: {exc}")

    answer = resp.choices[0].message.content or ""
    seen, sources = set(), []
    for i in top:
        key = (CHUNKS[i]["source"], CHUNKS[i]["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return {"answer": answer, "sources": sources}


@app.get("/")
def home():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Backend is running. The chat page comes in Step 3."}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")