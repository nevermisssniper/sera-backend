"""Service credentials and signed callbacks shared with the Next.js app."""
import hashlib
import hmac
import json
import os
import time
from urllib.parse import urlparse


def callback_config():
    url = os.environ.get("GENERATION_CALLBACK_URL", "")
    secret = os.environ.get("GENERATION_CALLBACK_SECRET", "")
    parsed = urlparse(url)
    if not secret or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("Configure GENERATION_CALLBACK_SECRET and an HTTPS GENERATION_CALLBACK_URL")
    return url, secret


def signed_callback(payload, secret):
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    timestamp = str(int(time.time()))
    signature = hmac.new(secret.encode("utf-8"), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return body, {
        "Content-Type": "application/json",
        "X-Generation-Timestamp": timestamp,
        "X-Generation-Signature": signature,
    }
