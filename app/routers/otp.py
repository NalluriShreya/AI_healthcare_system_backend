from fastapi import APIRouter, HTTPException, Request, status
from datetime import datetime, timedelta
from bson import ObjectId

from core.database import get_db
from core.security import hash_password, create_access_token
from core.utils import generate_otp, serialize_doc, hash_otp, check_rate_limit
from models.schemas import (
    OTPRequest, OTPVerify, TokenResponse,
    ForgotPasswordRequest, VerifyResetOTPRequest, ResetPasswordRequest
)
from services.email import send_otp_email

router = APIRouter()


# ── Phone OTP Login ────────────────────────────────────────────────────────────

@router.post("/otp/request")
async def request_otp(otp_request: OTPRequest):
    db = get_db()
    collection = db.patients if otp_request.role == "patient" else db.doctors
    user = await collection.find_one({"phone": otp_request.phone})

    if not user:
        raise HTTPException(status_code=404, detail=f"No {otp_request.role} found with this phone number")

    if otp_request.role == "patient" and not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Patient account is inactive")
    if otp_request.role == "doctor" and user.get("status") != "active":
        raise HTTPException(status_code=403, detail="Doctor account is not active")

    existing_otp = await db.otps.find_one({
        "user_id": user["_id"],
        "role": otp_request.role,
        "is_used": False,
        "expires_at": {"$gt": datetime.utcnow()}
    })

    if existing_otp:
        raise HTTPException(status_code=429, detail="OTP already sent. Please wait before requesting again.")

    otp_code = generate_otp()
    await db.otps.insert_one({
        "user_id": user["_id"],
        "role": otp_request.role,
        "otp_code": otp_code,
        "is_used": False,
        "expires_at": datetime.utcnow() + timedelta(minutes=5),
        "created_at": datetime.utcnow()
    })

    return {"message": "OTP sent successfully", "otp": otp_code, "expires_in_minutes": 5}


@router.post("/otp/verify", response_model=TokenResponse)
async def verify_otp(otp_verify: OTPVerify):
    db = get_db()
    collection = db.patients if otp_verify.role == "patient" else db.doctors
    user = await collection.find_one({"phone": otp_verify.phone})

    if not user:
        raise HTTPException(status_code=404, detail=f"No {otp_verify.role} found with this phone number")
    if otp_verify.role == "patient" and not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Patient account is inactive")
    if otp_verify.role == "doctor" and user.get("status") != "active":
        raise HTTPException(status_code=403, detail="Doctor account is not active")

    otp_record = await db.otps.find_one({
        "user_id": user["_id"],
        "role": otp_verify.role,
        "otp_code": otp_verify.otp_code,
        "is_used": False,
        "expires_at": {"$gt": datetime.utcnow()}
    })

    if not otp_record:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    await db.otps.update_one({"_id": otp_record["_id"]}, {"$set": {"is_used": True}})
    await collection.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})

    access_token = create_access_token(data={"sub": str(user["_id"]), "role": otp_verify.role})

    return {"access_token": access_token, "token_type": "bearer", "user": serialize_doc(user)}


# ── Forgot Password ────────────────────────────────────────────────────────────

@router.post("/auth/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest, request: Request):
    db = get_db()
    check_rate_limit(f"forgot:{payload.email}")

    if payload.role == "patient":
        collection = db.patients
    elif payload.role == "doctor":
        collection = db.doctors
    else:
        collection = db.admins

    user = await collection.find_one({"email": payload.email})
    GENERIC_MSG = "If this email is registered, you'll receive an OTP shortly."

    if not user:
        return {"message": GENERIC_MSG}
    if payload.role == "patient" and not user.get("is_active", True):
        return {"message": GENERIC_MSG}
    if payload.role == "doctor" and user.get("status") not in ("active",):
        return {"message": GENERIC_MSG}
    if payload.role == "admin" and not user.get("is_active", True):
        return {"message": GENERIC_MSG}

    await db.password_reset_otps.update_many(
        {"user_id": str(user["_id"]), "role": payload.role, "is_used": False},
        {"$set": {"is_used": True, "invalidated": True}}
    )

    otp_code = generate_otp()
    otp_hash = hash_otp(otp_code)

    await db.password_reset_otps.insert_one({
        "user_id": str(user["_id"]),
        "role": payload.role,
        "email": payload.email,
        "otp_hash": otp_hash,
        "is_used": False,
        "attempts": 0,
        "expires_at": datetime.utcnow() + timedelta(minutes=10),
        "created_at": datetime.utcnow()
    })

    await send_otp_email(payload.email, otp_code, user.get("name", "User"))
    return {"message": GENERIC_MSG}


@router.post("/auth/verify-reset-otp")
async def verify_reset_otp(payload: VerifyResetOTPRequest):
    db = get_db()
    otp_hash = hash_otp(payload.otp_code)

    otp_record = await db.password_reset_otps.find_one({
        "email": payload.email,
        "role": payload.role,
        "is_used": False,
        "expires_at": {"$gt": datetime.utcnow()}
    })

    if not otp_record:
        raise HTTPException(400, "Invalid or expired OTP. Please request a new one.")

    MAX_ATTEMPTS = 5
    await db.password_reset_otps.update_one({"_id": otp_record["_id"]}, {"$inc": {"attempts": 1}})

    if otp_record["attempts"] + 1 >= MAX_ATTEMPTS and otp_record["otp_hash"] != otp_hash:
        await db.password_reset_otps.update_one({"_id": otp_record["_id"]}, {"$set": {"is_used": True}})
        raise HTTPException(400, "Maximum OTP attempts exceeded. Please request a new OTP.")

    if otp_record["otp_hash"] != otp_hash:
        remaining = MAX_ATTEMPTS - (otp_record["attempts"] + 1)
        raise HTTPException(400, f"Incorrect OTP. {remaining} attempt(s) remaining.")

    return {"message": "OTP verified successfully.", "verified": True}


@router.post("/auth/reset-password")
async def reset_password(payload: ResetPasswordRequest):
    db = get_db()
    otp_hash = hash_otp(payload.otp_code)

    otp_record = await db.password_reset_otps.find_one({
        "email": payload.email,
        "role": payload.role,
        "otp_hash": otp_hash,
        "is_used": False,
        "expires_at": {"$gt": datetime.utcnow()}
    })

    if not otp_record:
        raise HTTPException(400, "Invalid or expired OTP. Please restart the process.")
    if otp_record.get("attempts", 0) >= 5:
        raise HTTPException(400, "Maximum attempts exceeded. Please request a new OTP.")

    if payload.role == "patient":
        collection = db.patients
    elif payload.role == "doctor":
        collection = db.doctors
    else:
        collection = db.admins

    result = await collection.update_one(
        {"_id": ObjectId(otp_record["user_id"])},
        {"$set": {"password_hash": hash_password(payload.new_password), "updated_at": datetime.utcnow()}}
    )

    if result.matched_count == 0:
        raise HTTPException(404, "User not found.")

    await db.password_reset_otps.update_one(
        {"_id": otp_record["_id"]},
        {"$set": {"is_used": True, "used_at": datetime.utcnow()}}
    )

    return {"message": "Password reset successfully. Please log in with your new password."}