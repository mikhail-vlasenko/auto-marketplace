"""fastapi_marketplace_backend.py

Chat‑based ad creation via OpenAI (`gpt-4-vision-preview` and `gpt-4`), single /chat-message endpoint.
Images are embedded in HTML `<img src="data:image/..."/>` within the prompt.
Uses Redis for chat history & pending drafts.
"""
from __future__ import annotations
import base64, json, os, uuid, logging, asyncio
from typing import List, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
import redis.asyncio as redis
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pprint import pprint
# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Settings & Redis
# --------------------------------------------------------------------------- #
class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    VLM_MODEL: str = os.getenv("VLM_MODEL", "gpt-4o")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    MARKETPLACE_API_URL: str = os.getenv("MARKETPLACE_API_URL", "https://api.marketplace.com/v1/listings")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TOP_P: float = float(os.getenv("TOP_P", "0.70"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    API_TOKEN: str = os.getenv("API_TOKEN", "")

settings = Settings()
logger.info("Loaded settings: %s", {k:v for k,v in settings.__dict__.items() if not k.startswith('_') and 'KEY' not in k.upper()})

# Initialize OpenAI client
openai_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=settings.REQUEST_TIMEOUT
)

redis_pool = redis.from_url(settings.REDIS_URL, decode_responses=False)
logger.info("Connected to Redis at %s", settings.REDIS_URL)

app = FastAPI(title="Chat‑Ad Backend", version="3.1.0")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    if not settings.API_TOKEN:
        logger.error("API_TOKEN not set in environment variables")
        raise HTTPException(status_code=500, detail="API token not configured")
    
    if credentials.credentials != settings.API_TOKEN:
        logger.warning("Invalid API token provided")
        raise HTTPException(status_code=401, detail="Invalid API token")
    
    return credentials.credentials

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ChatResponse(BaseModel):
    reply: str
    status: str               # "pending_info" | "posted"
    listing_id: Optional[str] = None

class NegotiationResponse(BaseModel):
    reply: str
    status: str = "negotiating"  # "negotiating" | "accepted" | "rejected"

class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    version: str

# --------------------------------------------------------------------------- #
# OpenAI helper
# --------------------------------------------------------------------------- #
async def _openai_chat(
    model: str,
    messages: List[dict],
    stream: bool = False
) -> str:
    logger.info("Making OpenAI API call with model %s", model)
    pprint(messages)
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            stream=stream,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("OpenAI API call failed: %s", str(e))
        raise

# --------------------------------------------------------------------------- #
# Vision & LLM
# --------------------------------------------------------------------------- #
def _encode_image(b: bytes) -> str:
    """Encode image bytes to base64 string for OpenAI API."""
    return base64.b64encode(b).decode('utf-8')

async def _vlm_extract_title_desc(images: List[bytes]) -> tuple[str,str]:
    logger.info("Extracting title and description from image")
    content_prompt = (
        "Create a marketplace listing from this image.\n"
        "1. Give a concise TITLE (≤12 words).\n"
        "2. Give a DESCRIPTION (2–4 sentences) with key features & condition.\n"
        "Respond with the title, then an empty line and then the description, which may span multiple lines. No other text.\n"
    )
    messages = [
        {
            "role": "user",
            "content": "What's in this image? Create a marketplace listing."
        },
        {
            "role": "assistant",
            "content": "I'll help you analyze this image and create a listing."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": content_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_encode_image(images[0])}"
                    }
                }
            ]
        }
    ]
    resp = await _openai_chat(settings.VLM_MODEL, messages)
    title, desc = resp.split("\n\n", 1)
    return title.strip(), desc.strip()

async def _llm_missing_info_question(title: str, desc: str, chat_history: List[dict]) -> Optional[str]:
    """Check if more information is needed for price estimation.
    
    Args:
        title: The listing title
        desc: The current description
        chat_history: List of previous chat messages with their responses
    """
    # Count existing questions
    question_count = sum(1 for msg in chat_history if msg.get("type") == "price_info_question")
    if question_count >= 2:
        return None

    # Build conversation with single system message
    messages = [
        {
            "role": "system",
            "content": "You are helping gather information to price a marketplace listing. Ask specific questions about condition, age, features, or usage - but never about price directly. If you have enough information or already asked 2 questions, respond with EXACTLY 'NONE'."
        }
    ]

    # Add listing info
    messages.append({
        "role": "user",
        "content": f"I want to sell this item:\nTITLE: {title}\nDESCRIPTION: {desc}"
    })

    # Add previous Q&A pairs in sequence
    for msg in chat_history:
        if msg.get("type") == "price_info_question":
            messages.append({"role": "assistant", "content": msg["text"]})
        elif msg.get("type") == "price_info_answer":
            messages.append({"role": "user", "content": msg["text"]})

    # Add final question
    messages.append({
        "role": "user",
        "content": "Do you need any more details to estimate the price? If yes, ask ONE specific question. If no, reply 'NONE'."
    })

    ans = await _openai_chat(settings.LLM_MODEL, messages)
    ans = ans.strip()
    return None if ans.upper()=="NONE" else ans

async def _append_price_info_qa(user_id: str, question: str, answer: Optional[str] = None):
    """Store a price information question and optionally its answer."""
    if answer is None:
        # Store question
        await redis_pool.rpush(f"chat:{user_id}", json.dumps({
            "from": "bot",
            "text": question,
            "type": "price_info_question"
        }))
    else:
        # Store answer
        await redis_pool.rpush(f"chat:{user_id}", json.dumps({
            "from": "user",
            "text": answer,
            "type": "price_info_answer"
        }))

async def _estimate_price(title: str, desc: str) -> float:
    """Estimate a fair market price for the item."""
    prompt = (
        "You are a marketplace pricing expert. Based on the item details below, estimate a fair market price in EURO.\n"
        "Consider factors like condition, features, and market value.\n"
        "Respond with ONLY a number (no currency symbol or text).\n"
        f"\nTITLE: {title}\nDESCRIPTION: {desc}"
    )
    messages = [{"role":"user","content": prompt}]
    try:
        price_str = await _openai_chat(settings.LLM_MODEL, messages)
        # Clean up response to get just the number
        price_str = ''.join(c for c in price_str if c.isdigit() or c == '.')
        return float(price_str)
    except (ValueError, TypeError) as e:
        logger.error("Failed to parse price estimation: %s", str(e))
        return 0.0

async def _post_listing(title: str, desc: str, images: List[bytes], price: float) -> str:
    logger.info("Posting new listing with title: %s", title)
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
        files = {f"image{idx}": (f"img{idx}.jpg", img, "image/jpeg")
                 for idx,img in enumerate(images)}
        try:
            logging.info(f"title: {title}")
            logging.info(f"desc: {desc}")
            logging.info(f"price: {price}")
            # r = await client.post(
            #     settings.MARKETPLACE_API_URL,
            #     data={"title":title, "description":desc, "price":price},
            #     files=files
            # )
            # r.raise_for_status()
            # listing_id = r.json().get("id", "")
            listing_id = "1234567890"
            logger.info("Successfully posted listing with ID: %s", listing_id)
            return listing_id
        except Exception as e:
            logger.error("Failed to post listing: %s", str(e))
            raise

# --------------------------------------------------------------------------- #
# Redis utils
# --------------------------------------------------------------------------- #
async def _save_pending(uid: str, d: dict): await redis_pool.set(f"ad:{uid}", json.dumps(d))
async def _load_pending(uid: str) -> Optional[dict]:
    v = await redis_pool.get(f"ad:{uid}"); return None if v is None else json.loads(v)
async def _clear_pending(uid: str): await redis_pool.delete(f"ad:{uid}")
async def _append_chat(uid: str, frm: str, txt: str):
    await redis_pool.rpush(f"chat:{uid}", json.dumps({"from":frm,"text":txt}))

async def _load_chat_history(uid: str) -> List[dict]:
    """Load all chat messages for a given conversation ID."""
    messages = await redis_pool.lrange(f"chat:{uid}", 0, -1)
    return [json.loads(m) for m in messages]

async def _save_listing(seller_id: str, title: str, desc: str, price: float, images: List[bytes]):
    """Store listing information in Redis. The listing ID is the same as the seller ID."""
    listing_data = {
        "title": title,
        "description": desc,
        "price": price,
        "images": [_encode_image(img) for img in images]
    }
    await redis_pool.set(f"listing:{seller_id}", json.dumps(listing_data))

async def _get_listing(seller_id: str) -> Optional[dict]:
    """Retrieve listing information from Redis using seller ID as the listing ID."""
    data = await redis_pool.get(f"listing:{seller_id}")
    return json.loads(data) if data else None

async def _delete_user_data(uid: str):
    """Delete all Redis records associated with a user."""
    # Delete chat history
    await redis_pool.delete(f"chat:{uid}")
    # Delete pending drafts
    await redis_pool.delete(f"ad:{uid}")
    # Delete user's listing
    await redis_pool.delete(f"listing:{uid}")
    
    # Delete all negotiation conversations where user is seller or buyer
    pattern = f"neg:{uid}:*"
    keys = await redis_pool.keys(pattern)
    if keys:
        await redis_pool.delete(*keys)
        # Delete associated chat histories
        for key in keys:
            # Extract buyer_id from neg:seller_id:buyer_id pattern
            buyer_id = key.split(":")[-1]
            await redis_pool.delete(f"chat:neg:{uid}:{buyer_id}")
    
    pattern = f"neg:*:{uid}"
    keys = await redis_pool.keys(pattern)
    if keys:
        await redis_pool.delete(*keys)
        # Delete associated chat histories
        for key in keys:
            # Extract seller_id from neg:seller_id:buyer_id pattern
            seller_id = key.split(":")[1]
            await redis_pool.delete(f"chat:neg:{seller_id}:{uid}")

# --------------------------------------------------------------------------- #
# Chat endpoint
# --------------------------------------------------------------------------- #
@app.post("/chat-message", response_model=ChatResponse)
async def chat_message(
    user_id: str = Form(...),
    text: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None),
    token: str = Depends(verify_token)
):
    logger.info("Received chat message from user %s", user_id)
    if not text and not images:
        logger.warning("Request missing both text and images")
        raise HTTPException(400, "Need text or images.")
    
    # log user
    if text: 
        logger.debug("User %s sent text: %s", user_id, text)
        await _append_chat(user_id, "user", text)
    
    img_bytes = []
    if images:
        for f in images:
            if f.content_type not in ("image/jpeg","image/png"):
                logger.warning("User %s attempted to upload unsupported file type: %s", user_id, f.content_type)
                raise HTTPException(400, "Only JPEG/PNG supported.")
            b=await f.read(); img_bytes.append(b)
        logger.info("User %s uploaded %d images", user_id, len(img_bytes))
        await _append_chat(user_id, "user", f"<sent {len(img_bytes)} images>")

    draft = await _load_pending(user_id)
    # first image triggers draft
    if img_bytes and not draft:
        ad_id = str(uuid.uuid4())
        title, desc = await _vlm_extract_title_desc(img_bytes)
        # Load chat history for context
        chat_history = await _load_chat_history(user_id)
        question = await _llm_missing_info_question(title, desc, chat_history)
        # save without price initially
        await _save_pending(user_id, {
            "ad_id": ad_id,
            "images": [_encode_image(b) for b in img_bytes],
            "title": title,
            "description": desc,
            "questions_asked": 0  # Track number of questions asked
        })
        if question:
            await _append_price_info_qa(user_id, question)
            return ChatResponse(reply=question, status="pending_info")
        
        # If no questions needed, estimate price and post
        price = await _estimate_price(title, desc)
        await _post_listing(title, desc, img_bytes, price)
        await _save_listing(user_id, title, desc, price, img_bytes)
        await _append_chat(user_id, "bot", f"Ad posted")
        await _clear_pending(user_id)
        return ChatResponse(reply="Ad posted successfully", status="posted", listing_id=user_id)

    # text follow-up with additional info
    if text and draft:
        # Store the answer to the previous question
        await _append_price_info_qa(user_id, None, text)
        
        # Append new info to description
        full_desc = draft["description"] + "\n\n" + text.strip()
        imgs = [base64.b64decode(x) for x in draft["images"]]
        
        # Check if we need more info, including chat history
        chat_history = await _load_chat_history(user_id)
        question = await _llm_missing_info_question(draft["title"], full_desc, chat_history)
        if question:
            # Update draft with new description
            draft["description"] = full_desc
            await _save_pending(user_id, draft)
            await _append_price_info_qa(user_id, question)
            return ChatResponse(reply=question, status="pending_info")
        
        # All info gathered, estimate price and post
        price = await _estimate_price(draft["title"], full_desc)
        await _post_listing(draft["title"], full_desc, imgs, price)
        await _save_listing(user_id, draft["title"], full_desc, price, imgs)
        await _append_chat(user_id, "bot", f"Ad posted")
        await _clear_pending(user_id)
        return ChatResponse(reply="Ad posted successfully", status="posted", listing_id=user_id)

    raise HTTPException(400, "No valid operation.")

@app.post("/negotiate", response_model=NegotiationResponse)
async def negotiate(
    seller_id: str = Form(...),
    buyer_id: str = Form(...),
    message: str = Form(...),
    token: str = Depends(verify_token)
):
    """Handle negotiation messages between buyer and seller."""
    logger.info("Received negotiation message from buyer %s to seller %s", 
                buyer_id, seller_id)
    
    # Create a unique conversation ID combining seller and buyer IDs
    conv_id = f"neg:{seller_id}:{buyer_id}"
    
    # Load the listing information using seller_id as listing_id
    listing = await _get_listing(seller_id)
    if not listing:
        raise HTTPException(404, "Seller has no active listing")
    
    # Load the negotiation conversation history
    negotiation_history = await _load_chat_history(conv_id)
    
    # Construct the prompt with full context
    system_prompt = (
        "You are negotiating with a potential buyer on behalf of your client, the seller. "
        "You want the best deal possible for your client. "
        "This is not the only potential buyer, so it is not critical to close the deal. "
        "Use the listing details and conversation history to help negotiate. "
        "Reply with short messages, no one wants to read long texts.\n\n"
    )

    # Add listing details to the prompt
    system_prompt += (
        f"LISTING DETAILS:\n"
        f"Title: {listing['title']}\n"
        f"Description: {listing['description']}\n"
        f"Listed Price: ${listing['price']:.2f}\n\n"
    )
    
    # Build the conversation context
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add negotiation history
    for msg in negotiation_history:
        messages.append({
            "role": "user" if msg["from"] == "buyer" else "assistant",
            "content": msg["text"]
        })
    
    # Add the new message
    messages.append({"role": "user", "content": message})
    
    # Get AI response
    try:
        response = await _openai_chat(settings.LLM_MODEL, messages)
        
        # Save the conversation
        await _append_chat(conv_id, "buyer", message)
        await _append_chat(conv_id, "bot", response)
        
        # Determine negotiation status based on response content
        status = "negotiating"
        if "accept" in response.lower() or "agreed" in response.lower():
            status = "accepted"
        elif "reject" in response.lower() or "declined" in response.lower():
            status = "rejected"
        
        return NegotiationResponse(reply=response, status=status)
        
    except Exception as e:
        logger.error("Failed to process negotiation: %s", str(e))
        raise HTTPException(500, "Failed to process negotiation message")

@app.delete("/user/{user_id}", response_model=dict)
async def delete_user_data(
    user_id: str,
    token: str = Depends(verify_token)
):
    """Delete all data associated with a user."""
    try:
        logger.info("Deleting all data for user %s", user_id)
        await _delete_user_data(user_id)
        return {"status": "success", "message": f"All data deleted for user {user_id}"}
    except Exception as e:
        logger.error("Failed to delete user data: %s", str(e))
        raise HTTPException(500, "Failed to delete user data")

# --------------------------------------------------------------------------- #
# Health endpoint
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health including Redis connection."""
    redis_ok = False
    try:
        # Test Redis connection with a simple ping
        await redis_pool.ping()
        redis_ok = True
    except Exception as e:
        logger.error("Redis health check failed: %s", str(e))
    
    return HealthResponse(
        status="healthy" if redis_ok else "degraded",
        redis_connected=redis_ok,
        version=app.version
    )
