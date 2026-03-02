from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from bson import ObjectId

from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.core.utils import serialize_doc
from app.models.schemas import (
    PatientSignup, PatientLogin,
    DoctorSignup, DoctorLogin,
    AdminLogin, TokenResponse
)

router = APIRouter()


# ── Patient Auth ───────────────────────────────────────────────────────────────

@router.post("/patient/signup", response_model=TokenResponse)
async def patient_signup(patient: PatientSignup):
    db = get_db()

    if await db.patients.find_one({"email": patient.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if await db.doctors.find_one({"email": patient.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if await db.admins.find_one({"email": patient.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if await db.patients.find_one({"phone": patient.phone}):
        raise HTTPException(status_code=400, detail="Phone number already registered")
    if await db.doctors.find_one({"phone": patient.phone}):
        raise HTTPException(status_code=400, detail="Phone number already registered")

    patient_doc = {
        "name": patient.name,
        "email": patient.email,
        "phone": patient.phone,
        "password_hash": hash_password(patient.password),
        "role": "patient",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow()
    }

    result = await db.patients.insert_one(patient_doc)
    access_token = create_access_token(data={"sub": str(result.inserted_id), "role": "patient"})
    user_data = serialize_doc(await db.patients.find_one({"_id": result.inserted_id}))

    return {"access_token": access_token, "token_type": "bearer", "user": user_data}


@router.post("/patient/login", response_model=TokenResponse)
async def patient_login(credentials: PatientLogin):
    db = get_db()
    patient = await db.patients.find_one({"email": credentials.email})

    if not patient:
        raise HTTPException(status_code=404, detail="No patient account found with this email")
    if not verify_password(credentials.password, patient["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect password")
    if not patient.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is inactive")

    await db.patients.update_one({"_id": patient["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    access_token = create_access_token(data={"sub": str(patient["_id"]), "role": "patient"})

    return {"access_token": access_token, "token_type": "bearer", "user": serialize_doc(patient)}


# ── Doctor Auth ────────────────────────────────────────────────────────────────

@router.post("/doctor/signup", response_model=TokenResponse)
async def doctor_signup(doctor: DoctorSignup):
    db = get_db()
    doctor_record = await db.doctors.find_one({"email": doctor.email})

    if not doctor_record:
        raise HTTPException(status_code=404, detail="No doctor account found with this email. Please contact admin.")
    if doctor_record.get("status") == "inactive":
        raise HTTPException(status_code=403, detail="Doctor account is inactive. Please contact admin.")
    if doctor_record.get("password_hash") is not None:
        raise HTTPException(status_code=400, detail="This account has already been activated")
    if doctor_record.get("phone") != doctor.phone:
        raise HTTPException(status_code=400, detail="Phone number does not match admin records")

    await db.doctors.update_one(
        {"_id": doctor_record["_id"]},
        {"$set": {
            "password_hash": hash_password(doctor.password),
            "status": "active",
            "last_login": datetime.utcnow()
        }}
    )

    access_token = create_access_token(data={"sub": str(doctor_record["_id"]), "role": "doctor"})
    user_data = serialize_doc(await db.doctors.find_one({"_id": doctor_record["_id"]}))

    return {"access_token": access_token, "token_type": "bearer", "user": user_data}


@router.post("/doctor/login", response_model=TokenResponse)
async def doctor_login(credentials: DoctorLogin):
    db = get_db()
    doctor = await db.doctors.find_one({"email": credentials.email})

    if not doctor:
        raise HTTPException(status_code=404, detail="No doctor account found. Please contact admin.")
    if not doctor.get("password_hash"):
        raise HTTPException(status_code=403, detail="Account not activated. Please complete signup first.")
    if not verify_password(credentials.password, doctor["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect password")
    if doctor.get("status") == "inactive":
        raise HTTPException(status_code=403, detail="Doctor account is inactive. Please contact admin.")
    if doctor.get("status") == "pending":
        raise HTTPException(status_code=400, detail="Please signup first")

    await db.doctors.update_one({"_id": doctor["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    access_token = create_access_token(data={"sub": str(doctor["_id"]), "role": "doctor"})

    return {"access_token": access_token, "token_type": "bearer", "user": serialize_doc(doctor)}


# ── Admin Auth ─────────────────────────────────────────────────────────────────

@router.post("/admin/login", response_model=TokenResponse)
async def admin_login(credentials: AdminLogin):
    db = get_db()
    admin = await db.admins.find_one({"email": credentials.email})

    if not admin:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(credentials.password, admin["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect password")
    if not admin.get("is_active", True):
        raise HTTPException(status_code=403, detail="Admin account is inactive")

    await db.admins.update_one({"_id": admin["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    access_token = create_access_token(data={"sub": str(admin["_id"]), "role": "admin"})

    return {"access_token": access_token, "token_type": "bearer", "user": serialize_doc(admin)}