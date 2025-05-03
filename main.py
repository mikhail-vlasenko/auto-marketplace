"""fastapi_marketplace_backend.py

Chat‑based ad creation via NVIDIA NIM (`google/gemma-3-27b-it`), single /chat-message endpoint.
Images are embedded in HTML `<img src="data:image/..."/>` within the prompt.
Uses Redis for chat history & pending drafts.
"""
from __future__ import annotations
import base64, json, os, uuid
from typing import List, Optional
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Settings & Redis
# --------------------------------------------------------------------------- #
class Settings:
    NIM_API_KEY: str = os.getenv("NIM_API_KEY", "")
    NIM_API_URL: str = os.getenv("NIM_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
    VLM_MODEL: str = os.getenv("VLM_MODEL", "google/gemma-3-27b-it")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "google/gemma-3-27b-it")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    MARKETPLACE_API_URL: str = os.getenv("MARKETPLACE_API_URL", "https://api.marketplace.com/v1/listings")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TOP_P: float = float(os.getenv("TOP_P", "0.70"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.20"))

settings = Settings()
redis_pool = redis.from_url(settings.REDIS_URL, decode_responses=False)
app = FastAPI(title="Chat‑Ad Backend", version="3.1.0")

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ChatResponse(BaseModel):
    reply: str
    status: str               # "pending_info" | "posted"
    listing_id: Optional[str] = None

# --------------------------------------------------------------------------- #
# NIM helper
# --------------------------------------------------------------------------- #
async def _nim_chat(
    model: str,
    messages: List[dict],
    stream: bool = False
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": settings.MAX_TOKENS,
        "temperature": settings.TEMPERATURE,
        "top_p": settings.TOP_P,
        "stream": stream,
    }
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
        resp = await client.post(
            settings.NIM_API_URL,
            headers={"Authorization": f"Bearer {settings.NIM_API_KEY}"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        # non-stream: single content
        return data["choices"][0]["message"]["content"]

# --------------------------------------------------------------------------- #
# Vision & LLM
# --------------------------------------------------------------------------- #
def _to_data_url(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()

async def _vlm_extract_title_desc(images: List[bytes]) -> tuple[str,str]:
    img_tag = f'<img src="{_to_data_url(images[0])}" />'
    content_prompt = (
        "Create a marketplace listing from this image.\n"
        "1. Give a concise TITLE (≤12 words).\n"
        "2. Give a DESCRIPTION (2–4 sentences) with key features & condition.\n"
        "Respond with the title and description only, no other text.\n"
    )
    messages = [{"role":"user","content": content_prompt + " " + img_tag}]
    resp = await _nim_chat(settings.VLM_MODEL, messages)
    return resp, resp

async def _llm_missing_info_question(title: str, desc: str) -> Optional[str]:
    prompt = (
        "Review the draft listing. If ALL critical info is present, reply EXACTLY 'NONE'."
        f"\nTITLE: {title}\nDESCRIPTION: {desc}\n"
        "Otherwise, ask one short question (≤15 words) for missing details."
    )
    messages = [{"role":"user","content": prompt}]
    ans = await _nim_chat(settings.LLM_MODEL, messages)
    ans = ans.strip()
    return None if ans.upper()=="NONE" else ans

async def _post_listing(title: str, desc: str, images: List[bytes]) -> str:
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
        files = {f"image{idx}": (f"img{idx}.jpg", img, "image/jpeg")
                 for idx,img in enumerate(images)}
        r = await client.post(
            settings.MARKETPLACE_API_URL,
            data={"title":title, "description":desc},
            files=files
        )
        r.raise_for_status()
        return r.json().get("id", "")

# --------------------------------------------------------------------------- #
# Redis utils
# --------------------------------------------------------------------------- #
async def _save_pending(uid: str, d: dict): await redis_pool.set(f"ad:{uid}", json.dumps(d))
async def _load_pending(uid: str) -> Optional[dict]:
    v = await redis_pool.get(f"ad:{uid}"); return None if v is None else json.loads(v)
async def _clear_pending(uid: str): await redis_pool.delete(f"ad:{uid}")
async def _append_chat(uid: str, frm: str, txt: str):
    await redis_pool.rpush(f"chat:{uid}", json.dumps({"from":frm,"text":txt}))

# --------------------------------------------------------------------------- #
# Chat endpoint
# --------------------------------------------------------------------------- #
@app.post("/chat-message", response_model=ChatResponse)
async def chat_message(
    user_id: str = Form(...),
    text: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None)
):
    if not text and not images:
        raise HTTPException(400, "Need text or images.")
    # log user
    if text: await _append_chat(user_id, "user", text)
    img_bytes = []
    if images:
        for f in images:
            if f.content_type not in ("image/jpeg","image/png"):
                raise HTTPException(400, "Only JPEG/PNG supported.")
            b=await f.read(); img_bytes.append(b)
        await _append_chat(user_id, "user", f"<sent {len(img_bytes)} images>")

    draft = await _load_pending(user_id)
    # first image triggers draft
    if img_bytes and not draft:
        ad_id = str(uuid.uuid4())
        title,desc = await _vlm_extract_title_desc(img_bytes)
        question = await _llm_missing_info_question(title,desc)
        # save
        await _save_pending(user_id, {"ad_id":ad_id,
            "images":[base64.b64encode(b).decode() for b in img_bytes],
            "title":title,"description":desc
        })
        if question:
            await _append_chat(user_id,"bot",question)
            return ChatResponse(reply=question,status="pending_info")
        lid = await _post_listing(title,desc,img_bytes)
        await _append_chat(user_id,"bot",f"Ad posted: {lid}")
        await _clear_pending(user_id)
        return ChatResponse(reply=f"Ad posted with ID {lid}", status="posted", listing_id=lid)

    # text follow-up
    if text and draft:
        full_desc = draft["description"] + "\n\n" + text.strip()
        imgs = [base64.b64decode(x) for x in draft["images"]]
        lid = await _post_listing(draft["title"], full_desc, imgs)
        await _append_chat(user_id,"bot",f"Ad posted: {lid}")
        await _clear_pending(user_id)
        return ChatResponse(reply=f"Ad posted with ID {lid}", status="posted", listing_id=lid)

    raise HTTPException(400, "No valid operation.")
