"""fastapi_marketplace_backend.py

Single endpoint (`/chat-message`) for all chat-based ad creation:
 • Handles user messages with optional images.
 • Stores full chat in Redis under `chat:{user_id}` list.
 • On first message with images, generates draft title/description via NVIDIA NIM VLM.
 • Checks for missing info via LLM; if needed, asks follow‑up question.
 • On text‑only messages (answers), merges into draft and publishes the ad.

Environment vars:
  NIM_API_KEY, NIM_API_URL, VLM_MODEL, LLM_MODEL,
  REDIS_URL, MARKETPLACE_API_URL

Dependencies: fastapi, httpx, redis.asyncio, python-multipart, pydantic
"""
from __future__ import annotations
import base64
import json
import os
import uuid
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

settings = Settings()
redis_pool = redis.from_url(settings.REDIS_URL, decode_responses=False)

app = FastAPI(title="Chat-based Marketplace Ad Backend", version="3.0.0")

# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
class ChatResponse(BaseModel):
    reply: str
    status: str               # "pending_info" | "posted"
    listing_id: Optional[str] = None

# --------------------------------------------------------------------------- #
# NVIDIA NIM helper
# --------------------------------------------------------------------------- #
async def _nim_chat(model: str, messages: list[dict], temperature: float = 0.2) -> str:
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
        resp = await client.post(
            settings.NIM_API_URL,
            headers={"Authorization": f"Bearer {settings.NIM_API_KEY}"},
            json={"model": model, "messages": messages, "temperature": temperature},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

# --------------------------------------------------------------------------- #
# Vision & LLM routines
# --------------------------------------------------------------------------- #
def _img_to_data_url(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()

async def _vlm_extract_title_desc(images: List[bytes]) -> tuple[str, str]:
    prompt = (
        "Create a marketplace listing from the image.\n"
        "1. Provide a concise TITLE (≤12 words).\n"
        "2. Provide a DESCRIPTION in 2-4 sentences, covering key features & condition.\n"
        "Respond ONLY with minified JSON: {\"title\":<title>,\"description\":<description>}"
    )
    msgs = [{"role":"user","content":prompt},
           {"role":"user","content":[{"type":"image_url","image_url":_img_to_data_url(images[0])}]}]
    content = await _nim_chat(settings.VLM_MODEL, msgs, temperature=0.0)
    data = json.loads(content)
    return data["title"].strip(), data["description"].strip()

async def _llm_missing_info_question(title: str, desc: str) -> Optional[str]:
    prompt = (
        "Review the draft listing. If ALL critical info is present, reply EXACTLY 'NONE'."
        f"\nTITLE: {title}\nDESCRIPTION: {desc}\n"
        "Otherwise, ask one short question (≤15 words) for missing details."
    )
    ans = await _nim_chat(settings.LLM_MODEL, [{"role":"user","content":prompt}], temperature=0)
    ans = ans.strip()
    return None if ans.upper()=="NONE" else ans

async def _post_marketplace_listing(title: str, description: str, images: List[bytes]) -> str:
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
        files = {f"image{idx}": (f"img{idx}.jpg", img, "image/jpeg")
                 for idx, img in enumerate(images)}
        resp = await client.post(settings.MARKETPLACE_API_URL,
                                 data={"title":title,"description":description},
                                 files=files)
        resp.raise_for_status()
        return resp.json().get("id", "")

# --------------------------------------------------------------------------- #
# Redis helpers
# --------------------------------------------------------------------------- #
async def _save_pending(user_id: str, draft: dict):
    await redis_pool.set(f"ad:{user_id}", json.dumps(draft))
async def _load_pending(user_id: str) -> Optional[dict]:
    v = await redis_pool.get(f"ad:{user_id}"); return None if v is None else json.loads(v)
async def _clear_pending(user_id: str):
    await redis_pool.delete(f"ad:{user_id}")
async def _append_chat(user_id: str, sender: str, text: str):
    entry = json.dumps({"from":sender,"text":text})
    await redis_pool.rpush(f"chat:{user_id}", entry)

# --------------------------------------------------------------------------- #
# Single chat endpoint
# --------------------------------------------------------------------------- #
@app.post("/chat-message", response_model=ChatResponse)
async def chat_message(
    user_id: str = Form(...),
    text: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None),
):
    # Validate input
    if not text and not images:
        raise HTTPException(400, "Provide at least text or images.")
    # Save user message
    if text:
        await _append_chat(user_id, "user", text)
    img_bytes_list = []
    if images:
        for img in images:
            if img.content_type not in ("image/jpeg","image/png"):
                raise HTTPException(400,"Only JPEG/PNG images supported.")
            b = await img.read(); img_bytes_list.append(b)
        await _append_chat(user_id, "user", f"<sent {len(img_bytes_list)} image(s)>")

    draft = await _load_pending(user_id)
    # New ad start
    if img_bytes_list and draft is None:
        ad_id = str(uuid.uuid4())
        title, desc = await _vlm_extract_title_desc(img_bytes_list)
        question = await _llm_missing_info_question(title, desc)
        # store draft
        await _save_pending(user_id, {"ad_id":ad_id,
                                      "images":[base64.b64encode(b).decode() for b in img_bytes_list],
                                      "title":title, "description":desc})
        if question:
            await _append_chat(user_id, "bot", question)
            return ChatResponse(reply=question, status="pending_info")
        # immediate post
        listing_id = await _post_marketplace_listing(title, desc, img_bytes_list)
        await _append_chat(user_id, "bot", f"Your ad is posted: {listing_id}")
        await _clear_pending(user_id)
        return ChatResponse(reply=f"Ad posted with ID {listing_id}", status="posted", listing_id=listing_id)

    # Follow‑up answer
    if text and draft:
        # user provided missing info
        full_desc = draft["description"] + "\n\n" + text.strip()
        images_decoded = [base64.b64decode(b) for b in draft["images"]]
        listing_id = await _post_marketplace_listing(draft["title"], full_desc, images_decoded)
        await _append_chat(user_id, "bot", f"Ad posted with ID {listing_id}")
        await _clear_pending(user_id)
        return ChatResponse(reply=f"Ad posted with ID {listing_id}", status="posted", listing_id=listing_id)

    # No action
    raise HTTPException(400, "No valid operation for given input.")
