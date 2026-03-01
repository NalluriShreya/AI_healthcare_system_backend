import random
import string
import calendar
import hashlib
import time
from datetime import datetime, timedelta, date
from typing import Optional
from bson import ObjectId
from fastapi import HTTPException


def generate_otp() -> str:
    return ''.join(random.choices(string.digits, k=6))


def serialize_doc(doc):
    if doc:
        doc["_id"] = str(doc["_id"])
        if "password_hash" in doc:
            del doc["password_hash"]
        return doc
    return None


def is_sunday_date(d: date) -> bool:
    return d.weekday() == 6


def hash_otp(otp: str) -> str:
    """SHA-256 hash of the OTP (never store plain OTP)."""
    return hashlib.sha256(otp.encode()).hexdigest()


async def count_leaves_in_month(db, doctor_id: str, year: int, month: int) -> int:
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])

    leaves = await db.doctor_leaves.find({
        "doctor_id": doctor_id,
        "approval_status": {"$in": ["approved", "auto_approved"]},
        "$or": [
            {"start_date": {"$lte": last_day.isoformat()},
             "end_date": {"$gte": first_day.isoformat()}}
        ]
    }).to_list(None)

    total = 0
    for leave in leaves:
        start = max(
            datetime.strptime(leave["start_date"], "%Y-%m-%d").date(),
            first_day
        )
        end = min(
            datetime.strptime(leave["end_date"], "%Y-%m-%d").date(),
            last_day
        )
        d = start
        while d <= end:
            if not is_sunday_date(d):
                total += 1
            d += timedelta(days=1)

    return total


# ── Rate limiting ──────────────────────────────────────────────────────────────
_rate_limit_store: dict = {}
RATE_LIMIT_MAX = 3
RATE_LIMIT_WINDOW = 600  # 10 minutes


def check_rate_limit(key: str):
    """Raise 429 if key has exceeded RATE_LIMIT_MAX calls in the window."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    calls = _rate_limit_store.get(key, [])
    calls = [t for t in calls if t > window_start]
    if len(calls) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    calls.append(now)
    _rate_limit_store[key] = calls