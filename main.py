# main.py
# FastAPI native Gemini proxy with rotating keys + API-key vs OAuth handling
# pip install fastapi uvicorn httpx

import os
import time
import asyncio
import json
import random
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import Response, JSONResponse, StreamingResponse
import httpx
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

log = logging.getLogger("uvicorn")
APP = FastAPI(title="Native Gemini proxy (auth-mode auto-detect)")

# -------------------------
# Config
# -------------------------
VPN_PROXY_URL = ""  # general proxy if needed
BRIGHT_DATA_URL = os.getenv("BRIGHT_DATA_PROXY_URL") # e.g. brd.superproxy.io:33335
BRIGHT_DATA_USER = os.getenv("BRIGHT_DATA_USER")
BRIGHT_DATA_PASS = os.getenv("BRIGHT_DATA_PASS")

KEYS_FILE = "api_keys.txt" # api keys, one per line
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = "gemini_api_keys"

ADMIN_TOKEN = "changeme_local_only"
UPSTREAM_BASE_GEMINI = "https://generativelanguage.googleapis.com/v1beta"
BACKOFF_MIN = 5
BACKOFF_MAX = 600
DEBUG = False

# -------------------------
# Setup proxy from config
# -------------------------
def get_proxy_for_request() -> Optional[str]:
    """
    Returns a Bright Data proxy URL with a random session ID for IP rotation.
    If Bright Data is not configured, falls back to VPN_PROXY_URL.
    """
    if BRIGHT_DATA_URL and BRIGHT_DATA_USER and BRIGHT_DATA_PASS:
        # Generate a random session ID for each request to ensure IP rotation
        session_id = f"sess_{random.randint(10000, 99999)}"
        # Format: http://user-session-random:pass@host:port
        return f"http://{BRIGHT_DATA_USER}-session-{session_id}:{BRIGHT_DATA_PASS}@{BRIGHT_DATA_URL}"
    
    if VPN_PROXY_URL:
        return VPN_PROXY_URL if "://" in VPN_PROXY_URL else f"http://{VPN_PROXY_URL}"
    
    return None

# Keep environment variables for general tools if needed, but we'll use dynamic proxy in client
if VPN_PROXY_URL:
    proxy_url_with_scheme = VPN_PROXY_URL if "://" in VPN_PROXY_URL else f"http://{VPN_PROXY_URL}"
    os.environ['HTTP_PROXY'] = proxy_url_with_scheme
    os.environ['HTTPS_PROXY'] = proxy_url_with_scheme
    os.environ['ALL_PROXY'] = proxy_url_with_scheme

# -------------------------
# Utilities: load keys
# -------------------------
def load_keys_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        log.warning(f"API keys file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    return keys

def load_keys_from_supabase() -> List[str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        missing = []
        if not SUPABASE_URL: missing.append("SUPABASE_URL")
        if not SUPABASE_KEY: missing.append("SUPABASE_KEY")
        log.warning(f"Supabase credentials not configured. Missing: {', '.join(missing)}")
        return []
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Fetch active keys from Supabase
        query = supabase.table(SUPABASE_TABLE).select("*")
        
        # Check if 'is_active' column exists (optional but good practice)
        # For now, we'll just try to filter by it if it exists
        response = query.execute()
        
        if not response.data:
            log.warning(f"No data found in Supabase table: {SUPABASE_TABLE}")
            return []
        
        # Filter active ones if 'is_active' exists in the first row
        data = response.data
        if "is_active" in data[0]:
            data = [row for row in data if row.get("is_active") is True]
            
        if not data:
            log.warning(f"No active keys found in Supabase table: {SUPABASE_TABLE}")
            return []

        keys = []
        # Find column that looks like a key
        possible_cols = ["key", "api_key", "token", "value"]
        first_row = data[0]
        key_col = None
        for col in possible_cols:
            if col in first_row:
                key_col = col
                break
        
        if not key_col:
            # If no obvious key column, use the first column that has AIza... or looks like a key
            for col, val in first_row.items():
                if isinstance(val, str) and (val.startswith("AIza") or len(val) > 30):
                    key_col = col
                    break
        
        if key_col:
            keys = [row[key_col] for row in data if row.get(key_col)]
            log.info(f"Loaded {len(keys)} active keys from Supabase table '{SUPABASE_TABLE}' (column: '{key_col}')")
        else:
            log.warning(f"Could not identify key column in Supabase table: {SUPABASE_TABLE}. Available columns: {list(first_row.keys())}")
            
        return keys
    except Exception as e:
        log.error(f"Error loading keys from Supabase: {e}")
        return []

def get_all_keys() -> List[str]:
    file_keys = load_keys_from_file(KEYS_FILE)
    supabase_keys = load_keys_from_supabase()
    all_keys = list(set(file_keys + supabase_keys)) # Deduplicate
    if not all_keys:
        raise RuntimeError("No API keys found in file or Supabase.")
    log.info(f"Total keys loaded: {len(all_keys)} (File: {len(file_keys)}, Supabase: {len(supabase_keys)})")
    for i, k in enumerate(all_keys):
        log.info(f"Key {i+1}: {k[:10]}...")
    return all_keys

KEYS_LIST = get_all_keys()

# -------------------------
# Key state & pool (simple backoff-based)
# -------------------------
class KeyState:
    def __init__(self, key: str):
        self.key: str = key
        self.backoff: float = 0.0
        self.banned_until: float = 0.0
        self.success: int = 0
        self.fail: int = 0

    def is_available(self) -> bool:
        return time.monotonic() >= self.banned_until

    def mark_success(self) -> None:
        self.backoff = 0.0
        self.banned_until = 0.0
        self.success += 1

    def mark_failure(self) -> None:
        if self.backoff <= 0:
            self.backoff = BACKOFF_MIN
        else:
            self.backoff = min(BACKOFF_MAX, self.backoff * 2.0)
        self.banned_until = time.monotonic() + self.backoff
        self.fail += 1


class KeyPool:
    def __init__(self, keys: List[str]):
        self.states: List[KeyState] = [KeyState(k) for k in keys]
        self.n: int = len(self.states)
        self.idx: int = 0
        self.lock = asyncio.Lock()

    async def next_available(self) -> Optional[KeyState]:
        async with self.lock:
            start = self.idx
            for i in range(self.n):
                j = (start + i) % self.n
                st = self.states[j]
                if st.is_available():
                    self.idx = (j + 1) % self.n
                    return st
            return None

    def status(self) -> List[Dict[str, Any]]:
        now = time.monotonic()
        out: List[Dict[str, Any]] = []
        for s in self.states:
            out.append({
                "key_preview": (s.key[:12] + "...") if len(s.key) > 8 else s.key,
                "available_in": max(0, round(s.banned_until - now, 2)),
                "backoff": s.backoff,
                "success": s.success,
                "fail": s.fail,
            })
        return out


POOL = KeyPool(KEYS_LIST)

# -------------------------
# Routing helpers (fixed: avoid double v1/v1beta)
# -------------------------
def map_incoming_to_upstream(path: str) -> str:
    """
    Map incoming path -> native Gemini upstream URL or OpenAI-compatible URL.
    """
    p = path.lstrip("/")
    
    # Check if it's an OpenAI-style request
    if p.startswith("v1/chat/completions") or p.startswith("v1/models") or "chat/completions" in p or "models" in p:
        if p.startswith("v1/"):
            p = p[len("v1/"):]
        # Use Google's OpenAI-compatible endpoint
        return UPSTREAM_BASE_GEMINI.rstrip("/") + "/openai/v1/" + p

    # Original Gemini native logic
    if p.startswith("v1/"):
        p = p[len("v1/"):]
    elif p.startswith("v1beta/"):
        p = p[len("v1beta/"):]
    
    if p == "" or p == "v1":
        return UPSTREAM_BASE_GEMINI.rstrip("/")
    return UPSTREAM_BASE_GEMINI.rstrip("/") + "/" + p


def detect_stream_from_request(content_bytes: Optional[bytes], query_params: Dict[str, Any]) -> bool:
    # Gemini native streaming uses alt=sse
    if query_params.get("alt") == "sse":
        return True
    # Also support stream=true for compatibility with some clients
    qp = query_params.get("stream")
    if qp in ("true", "True", "1", True):
        return True
    if content_bytes:
        try:
            j = json.loads(content_bytes.decode(errors="ignore"))
            if isinstance(j, dict) and j.get("stream") is True:
                return True
        except Exception:
            pass
    return False


def prepare_auth_for_key(incoming_headers: Dict[str, str], incoming_params: Dict[str, Any], key_state: KeyState, is_openai: bool = False):
    """
    Return (headers_copy, params_copy) where authentication for key_state.key is applied.
    - If is_openai is True, always use Authorization: Bearer <key>.
    - If key looks like API key (starts with 'AIza') and not is_openai, put it as params['key'].
    - Otherwise set Authorization: Bearer <key>.
    """
    headers = dict(incoming_headers)
    params = dict(incoming_params) if incoming_params is not None else {}

    k = key_state.key.strip()
    
    if is_openai:
        # OpenAI-compatible endpoint expects Authorization: Bearer <key>
        headers['Authorization'] = f"Bearer {k}"
        auth_mode = "bearer_header(openai)"
    elif k.startswith("AIza"):
        # use query parameter 'key' for API key (do not set Authorization)
        params['key'] = k
        if 'authorization' in {x.lower() for x in headers.keys()}:
            # remove incoming Authorization to avoid confusion
            headers = {hk: hv for hk, hv in headers.items() if hk.lower() != 'authorization'}
        auth_mode = "api_key(query)"
    else:
        # assume OAuth access token / service account token etc.
        headers['Authorization'] = f"Bearer {k}"
        auth_mode = "bearer_header"
    
    if DEBUG:
        print(f"[DEBUG] auth mode {auth_mode} for key preview {k[:12]}...")
    return headers, params


# -------------------------
# Catch-all proxy endpoint
# -------------------------
@APP.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(request: Request, full_path: str):
    # Handle connection test from n8n/OpenAI clients
    p = full_path.lstrip("/")
    if p in ("v1", "v1/models", "models"):
        return JSONResponse({
            "object": "list",
            "data": [{"id": "gemini-1.5-flash", "object": "model", "owned_by": "google"}]
        })

    upstream_url = map_incoming_to_upstream(full_path)
    content = await request.body()
    params = dict(request.query_params)

    # copy incoming headers but skip hop-by-hop
    incoming_headers: Dict[str, str] = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding", "connection")
    }

    is_stream = detect_stream_from_request(content if content else None, params)

    # Gemini native streaming vs OpenAI streaming
    is_openai_path = "/openai/" in upstream_url
    if is_stream and not is_openai_path and ":generateContent" in upstream_url:
        upstream_url = upstream_url.replace(":generateContent", ":streamGenerateContent")
        if 'stream' in params:
            del params['stream']

    # --- Streaming requests ---
    if is_stream:
        async def stream_generator():
            tried_keys, logged_errors = [], []
            for _ in range(len(POOL.states)):
                key_state = await POOL.next_available()
                if not key_state: break
                tried_keys.append(key_state.key[:12] + "...")
                headers_auth, params_auth = prepare_auth_for_key(incoming_headers, params, key_state, is_openai=is_openai_path)
                if not any(k.lower() == "content-type" for k in headers_auth.keys()):
                    headers_auth["Content-Type"] = request.headers.get("content-type", "application/json")

                if DEBUG: print(f"[DEBUG] Attempting stream with key {key_state.key[:12]}...")
                
                proxy_url = get_proxy_for_request()
                proxies = {"all://": proxy_url} if proxy_url else None
                
                try:
                    async with httpx.AsyncClient(timeout=300, proxy=proxy_url) as client, client.stream(
                        request.method, upstream_url, headers=headers_auth, params=params_auth, content=content
                    ) as upstream:
                        if upstream.status_code >= 400:
                            key_state.mark_failure()
                            body = await upstream.aread()
                            logged_errors.append({"key": key_state.key[:12], "status": upstream.status_code, "body": body.decode(errors='ignore')})
                            log.warning(f"Key {key_state.key[:12]}... failed on stream connection with status {upstream.status_code}. Retrying...")
                            continue

                        is_first_chunk, stream_had_error = True, False
                        async for chunk in upstream.aiter_bytes():
                            if is_first_chunk:
                                is_first_chunk = False
                                # Gemini streams a 'data: ' prefix, which we can ignore for error checking
                                chunk_content_for_check = chunk
                                if chunk_content_for_check.startswith(b'data: '):
                                    chunk_content_for_check = chunk_content_for_check[len(b'data: '):]
                                
                                try:
                                    # The first chunk might be a list with a single error object
                                    data = json.loads(chunk_content_for_check.decode())
                                    if isinstance(data, list): data = data

                                    if isinstance(data, dict) and "error" in data:
                                        key_state.mark_failure()
                                        stream_had_error = True
                                        msg = data.get("error", {}).get("message", "Unknown stream error")
                                        logged_errors.append({"key": key_state.key[:12], "status": "in-stream", "body": msg})
                                        if DEBUG: print(f"[DEBUG] In-stream error for key {key_state.key[:12]}...: {msg}")
                                        break 
                                except (json.JSONDecodeError, UnicodeDecodeError, IndexError): pass
                            yield chunk
                        
                        if stream_had_error: continue
                        key_state.mark_success()
                        client_info = f" to {request.client.host}:{request.client.port}" if request.client else ""
                        log.info(f"Stream{client_info} completed successfully with key {key_state.key[:12]}...")
                        return
                except httpx.RequestError as e:
                    key_state.mark_failure()
                    logged_errors.append({"key": key_state.key[:12], "error": str(e)})
                    if DEBUG: print(f"[DEBUG] Request error for stream key {key_state.key[:12]}...: {e}")
                    continue
            
            if not tried_keys:
                log.error("All keys are within rate limit. Could not process stream request.")

            #FIXME: Roo Code doesn't understand this error
            final_error = {"error": {"code": 502, "message": "All keys failed for streaming request.", "details": logged_errors}}
            yield (f"data: {json.dumps(final_error)}\r\n\r\n").encode()
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    # --- Non-streaming requests ---
    else:
        tried, errors = [], []
        for _ in range(len(POOL.states)):
            key_state = await POOL.next_available()
            if not key_state: break
            tried.append(key_state.key[:12] + "...")
            headers_auth, params_auth = prepare_auth_for_key(incoming_headers, params, key_state, is_openai=is_openai_path)
            if not any(k.lower() == "content-type" for k in headers_auth.keys()):
                headers_auth["Content-Type"] = request.headers.get("content-type", "application/json")

            if DEBUG: print(f"[DEBUG] trying key {key_state.key[:12]}... -> {upstream_url}")
            
            proxy_url = get_proxy_for_request()
            
            try:
                async with httpx.AsyncClient(timeout=300, proxy=proxy_url) as client:
                    resp = await client.request(request.method, upstream_url, headers=headers_auth, params=params_auth, content=content)
                
                if resp.status_code < 400:
                    key_state.mark_success()
                    client_info = f" from {request.client.host}:{request.client.port}" if request.client else ""
                    log.info(f"Request{client_info} completed successfully with key {key_state.key[:12]}...")
                    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

                # It's an error, mark failure
                key_state.mark_failure()
                error_body_str = resp.text
                log.error(f"Key {key_state.key[:12]}... failed with status {resp.status_code}, body: {error_body_str}")

                # Also treat 400 as retryable for cases like invalid API keys
                if resp.status_code in (400, 429, 500, 502, 503):
                    errors.append({"key_preview": key_state.key[:12] + "...", "error": error_body_str, "status_code": resp.status_code})
                    continue # Retryable error, try next key
                else:
                    # Non-retryable error, return immediately
                    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

            except httpx.RequestError as e:
                key_state.mark_failure()
                errors.append({"key_preview": key_state.key[:12] + "...", "error": str(e)})
                if DEBUG: print(f"[DEBUG] Request error for key {key_state.key[:12]}...: {e}")
                continue

        if not tried:
            log.error("All keys are in backoff. Could not process request.")
            return JSONResponse({"error": "all keys rate-limited or in backoff"}, status_code=429)
        return JSONResponse({"error": "no upstream key succeeded", "tried": tried, "errors": errors}, status_code=502)


# -------------------------
# Admin endpoints
# -------------------------
def is_admin(auth_header: Optional[str]) -> bool:
    if not auth_header:
        return False
    if auth_header == ADMIN_TOKEN:
        return True
    low = auth_header.lower()
    if low.startswith("bearer "):
        return auth_header.split(" ", 1) == ADMIN_TOKEN
    return False


@APP.get("/status")
async def status(x_proxy_admin: Optional[str] = Header(None)):
    if not is_admin(x_proxy_admin):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return JSONResponse({"keys": POOL.status()})


@APP.get("/list-keys")
async def list_keys():
    """
    Public endpoint to verify keys without exposing full sensitive data.
    """
    out = []
    now = time.monotonic()
    for s in POOL.states:
        k = s.key.strip()
        # Show first 10 and last 5 chars for identification
        masked = f"{k[:10]}...{k[-5:]}" if len(k) > 15 else k
        out.append({
            "key_preview": masked,
            "is_available": s.is_available(),
            "available_in": max(0, round(s.banned_until - now, 2)),
            "success_count": s.success,
            "fail_count": s.fail,
        })
    return JSONResponse({
        "total_keys": len(out),
        "active_keys": len([x for x in out if x["is_available"]]),
        "keys": out
    })


@APP.post("/reload-keys")
async def reload_keys(x_proxy_admin: Optional[str] = Header(None)):
    if not is_admin(x_proxy_admin):
        raise HTTPException(status_code=401, detail="Unauthorized")
    global KEYS_LIST, POOL
    KEYS_LIST = get_all_keys()
    POOL = KeyPool(KEYS_LIST)
    return JSONResponse({"reloaded": True, "num_keys": len(KEYS_LIST)})


# -------------------------
# Run note:
# uvicorn main:APP --host 0.0.0.0 --port 3000
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(APP, host="0.0.0.0", port=port)
