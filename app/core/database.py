from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from typing import Optional
from app.core.config import MONGODB_URL, DATABASE_NAME

client: Optional[AsyncIOMotorClient] = None
db = None


def get_db():
    return db


async def connect_db():
    global client, db
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    # ===== UNIQUE INDEXES =====
    await db.patients.create_index("email", unique=True)
    await db.patients.create_index("phone", unique=True)

    await db.doctors.create_index("email", unique=True)
    await db.doctors.create_index("phone", unique=True)
    await db.doctors.create_index("doctor_id", unique=True)

    await db.admins.create_index("email", unique=True)

    # ===== OTP TTL =====
    await db.otps.create_index("expires_at", expireAfterSeconds=0)

    # ===== DOCTOR AVAILABILITY =====
    await db.doctor_availability.create_index(
        [("doctor_id", 1), ("date", 1)], unique=True
    )

    # ===== APPOINTMENTS =====
    await db.appointments.create_index(
        [("doctor_id", 1), ("date", 1), ("slot", 1)]
    )
    await db.appointments.create_index(
        [("patient_id", 1), ("date", -1)]
    )
    await db.appointments.create_index("status")

    # ===== LEAVE MANAGEMENT =====
    await db.doctor_leaves.create_index(
        [("doctor_id", 1), ("end_date", 1), ("start_date", 1)]
    )
    await db.doctor_leaves.create_index("approval_status")
    await db.doctor_leaves.create_index("created_by_admin_id")

    await db.appointments.create_index(
        [("date", 1), ("status", 1)]
    )

    # ===== FILTERING / SORTING =====
    await db.doctors.create_index("status")
    await db.doctors.create_index("department")
    await db.doctors.create_index([("created_at", DESCENDING)])
    await db.doctors.create_index([("updated_at", DESCENDING)])

    # ===== PATIENT NOTIFICATIONS =====
    await db.patient_notifications.create_index(
        [("patient_id", 1), ("created_at", -1)]
    )

    # DOCTOR NOTIFICATIONS
    await db.doctor_notifications.create_index(
        [("doctor_id", 1), ("created_at", -1)]
    )

    # Forgot Password OTP collection
    await db.password_reset_otps.create_index("expires_at", expireAfterSeconds=0)
    await db.password_reset_otps.create_index([("email", 1), ("role", 1)])
    await db.password_reset_otps.create_index("user_id")

    await db.accounts.create_index("transaction_id", unique=True)
    await db.accounts.create_index([("patient_id", 1), ("paid_at", -1)])
    await db.accounts.create_index([("doctor_id", 1), ("paid_at", -1)])
    await db.accounts.create_index("appointment_id")
    await db.accounts.create_index("status")


async def close_db():
    if client:
        client.close()