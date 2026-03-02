from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, date
from typing import Optional
from bson import ObjectId
import secrets

from app.core.database import get_db
from app.core.security import get_current_user, verify_password, hash_password
from app.core.utils import serialize_doc, is_sunday_date
from app.models.schemas import PatientProfileUpdate, AppointmentCreate
from app.services.notifications import (
    notify_appointment_status_change,
    notify_doctor_new_appointment,
    notify_doctor_appointment_cancelled
)

router = APIRouter(prefix="/api/patient", tags=["Patient"])


@router.get("/notifications")
async def get_patient_notifications(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "patient":
        raise HTTPException(403, "Patient access required")
    db = get_db()

    notifications = await db.patient_notifications.find(
        {"patient_id": current_user["user_id"]}
    ).sort("created_at", -1).limit(20).to_list(None)

    return {"notifications": [{**n, "_id": str(n["_id"])} for n in notifications]}


@router.get("/payment-history")
async def get_patient_payment_history(current_user: dict = Depends(get_current_user)):
    """Get patient's full payment history with receipt details"""
    if current_user["role"] != "patient":
        raise HTTPException(403, "Patient access required")
    db = get_db()
    patient_id = current_user["user_id"]

    payments = await db.accounts.find({"patient_id": patient_id}).sort("paid_at", -1).to_list(None)

    enriched = []
    for payment in payments:
        appointment = None
        if payment.get("appointment_id"):
            appointment = await db.appointments.find_one({"_id": ObjectId(payment["appointment_id"])})

        doctor = None
        if payment.get("doctor_id"):
            doctor = await db.doctors.find_one({"_id": ObjectId(payment["doctor_id"])})

        payment_data = {
            "_id": str(payment["_id"]),
            "transaction_id": payment.get("transaction_id", "N/A"),
            "appointment_id": payment.get("appointment_id"),
            "appointment_date": payment.get("appointment_date"),
            "slot": payment.get("slot"),
            "doctor_name": payment.get("doctor_name"),
            "doctor_department": doctor.get("department") if doctor else None,
            "doctor_specialization": doctor.get("specialization") if doctor else None,
            "doctor_qualification": doctor.get("qualification") if doctor else None,
            "consultation_fee": payment.get("consultation_fee", 299),
            "platform_fee": payment.get("platform_fee", 19),
            "payment_amount": payment.get("payment_amount", 318),
            "payment_method": payment.get("payment_method", "card"),
            "status": payment.get("status", "success"),
            "refund_amount": payment.get("refund_amount", 0),
            "refund_at": payment.get("refund_at"),
            "refund_transaction_id": payment.get("refund_transaction_id"),
            "refund_reason": payment.get("refund_reason"),
            "paid_at": payment.get("paid_at"),
            "created_at": payment.get("created_at"),
            "appointment_status": appointment.get("status") if appointment else None,
            "token_number": appointment.get("token_number") if appointment else None,
            "symptoms": appointment.get("symptoms") if appointment else None,
        }

        for date_field in ["refund_at", "paid_at", "created_at"]:
            if payment_data[date_field] and hasattr(payment_data[date_field], "isoformat"):
                payment_data[date_field] = payment_data[date_field].isoformat()

        enriched.append(payment_data)

    total_spent = sum(p["payment_amount"] for p in enriched if p["status"] == "success")
    total_refunded = sum(p.get("refund_amount", 0) for p in enriched if p["status"] == "refunded")
    successful = [p for p in enriched if p["status"] == "success"]
    refunded = [p for p in enriched if p["status"] == "refunded"]

    return {
        "payments": enriched,
        "summary": {
            "total_transactions": len(enriched),
            "total_spent": total_spent,
            "total_refunded": total_refunded,
            "net_spent": total_spent - total_refunded,
            "successful_count": len(successful),
            "refunded_count": len(refunded),
        }
    }


@router.patch("/profile")
async def update_patient_profile(
    payload: PatientProfileUpdate,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "patient":
        raise HTTPException(403, "Patient access required")
    db = get_db()
    patient_id = current_user["user_id"]
    patient = await db.patients.find_one({"_id": ObjectId(patient_id)})

    if not patient:
        raise HTTPException(404, "Patient not found")

    update_data = {}

    if payload.name is not None:
        update_data["name"] = payload.name

    if payload.email is not None and payload.email != patient["email"]:
        if await db.patients.find_one({"email": payload.email, "_id": {"$ne": ObjectId(patient_id)}}):
            raise HTTPException(400, "Email already in use")
        if await db.doctors.find_one({"email": payload.email}):
            raise HTTPException(400, "Email already in use")
        if await db.admins.find_one({"email": payload.email}):
            raise HTTPException(400, "Email already in use")
        update_data["email"] = payload.email

    if payload.phone is not None and payload.phone != patient["phone"]:
        if await db.patients.find_one({"phone": payload.phone, "_id": {"$ne": ObjectId(patient_id)}}):
            raise HTTPException(400, "Phone number already in use")
        if await db.doctors.find_one({"phone": payload.phone}):
            raise HTTPException(400, "Phone number already in use")
        if await db.admins.find_one({"phone": payload.phone}):
            raise HTTPException(400, "Phone number already in use")
        update_data["phone"] = payload.phone

    if payload.new_password:
        if not payload.current_password:
            raise HTTPException(400, "Current password is required to set a new password")
        if not verify_password(payload.current_password, patient["password_hash"]):
            raise HTTPException(400, "Current password is incorrect")
        update_data["password_hash"] = hash_password(payload.new_password)

    if not update_data:
        raise HTTPException(400, "No fields to update")

    update_data["updated_at"] = datetime.utcnow()
    await db.patients.update_one({"_id": ObjectId(patient_id)}, {"$set": update_data})

    propagate = {}
    if "name" in update_data:
        propagate["patient_name"] = update_data["name"]

    if propagate:
        await db.appointments.update_many({"patient_id": patient_id}, {"$set": propagate})
        await db.accounts.update_many({"patient_id": patient_id}, {"$set": propagate})

    updated = await db.patients.find_one({"_id": ObjectId(patient_id)})
    return {"message": "Profile updated successfully", "user": serialize_doc(updated)}


@router.get("/doctors/search")
async def search_available_doctors(
    date: str,
    department: Optional[str] = None,
    slot: Optional[str] = None,
    search: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "patient":
        raise HTTPException(status_code=403, detail="Patient access required")
    db = get_db()

    try:
        from datetime import date as dt_date
        search_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    if search_date < datetime.now().date():
        raise HTTPException(status_code=400, detail="Cannot book appointments for past dates")

    if is_sunday_date(search_date):
        raise HTTPException(status_code=400, detail="Sunday is not a working day. Please select another date.")

    doctor_query = {"status": "active"}
    if department:
        doctor_query["department"] = department
    if search and search.strip():
        doctor_query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"specialization": {"$regex": search, "$options": "i"}}
        ]

    doctors = await db.doctors.find(doctor_query).sort("name", 1).to_list(None)
    results = []

    for doctor in doctors:
        doctor_id = str(doctor["_id"])

        leave = await db.doctor_leaves.find_one({
            "doctor_id": doctor_id,
            "start_date": {"$lte": date},
            "end_date": {"$gte": date},
            "approval_status": {"$in": ["approved", "auto_approved"]}
        })
        if leave:
            continue

        availability = await db.doctor_availability.find_one({"doctor_id": doctor_id, "date": date})

        if not availability:
            availability = {
                "is_available": True,
                "morning_slot_enabled": True,
                "afternoon_slot_enabled": True,
                "morning_capacity": 100,
                "afternoon_capacity": 100
            }

        if not availability.get("is_available", True):
            continue

        morning_enabled = availability.get("morning_slot_enabled", True)
        afternoon_enabled = availability.get("afternoon_slot_enabled", True)

        if slot == "morning" and not morning_enabled:
            continue
        if slot == "afternoon" and not afternoon_enabled:
            continue

        morning_booked = await db.appointments.count_documents({
            "doctor_id": doctor_id, "date": date, "slot": "morning",
            "status": {"$in": ["confirmed", "pending"]}
        })
        afternoon_booked = await db.appointments.count_documents({
            "doctor_id": doctor_id, "date": date, "slot": "afternoon",
            "status": {"$in": ["confirmed", "pending"]}
        })

        morning_capacity = availability.get("morning_capacity", 100)
        afternoon_capacity = availability.get("afternoon_capacity", 100)

        doctor_data = serialize_doc(doctor)
        doctor_data["availability"] = {
            "morning": {
                "enabled": morning_enabled,
                "available": morning_enabled and morning_booked < morning_capacity,
                "booked": morning_booked,
                "capacity": morning_capacity,
                "remaining": max(morning_capacity - morning_booked, 0)
            },
            "afternoon": {
                "enabled": afternoon_enabled,
                "available": afternoon_enabled and afternoon_booked < afternoon_capacity,
                "booked": afternoon_booked,
                "capacity": afternoon_capacity,
                "remaining": max(afternoon_capacity - afternoon_booked, 0)
            }
        }
        results.append(doctor_data)

    return {"date": date, "total": len(results), "doctors": results}


@router.post("/appointment/book")
async def book_appointment(
    appointment: AppointmentCreate,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "patient":
        raise HTTPException(403, "Patient access required")
    db = get_db()
    patient_id = current_user["user_id"]

    try:
        apt_date = datetime.strptime(appointment.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Invalid date format")

    if is_sunday_date(apt_date):
        raise HTTPException(400, "Sunday is not a working day. Please select another date.")
    if apt_date < date.today():
        raise HTTPException(400, "Cannot book appointments for past dates")

    if apt_date == date.today():
        now = datetime.now()
        current_minutes = now.hour * 60 + now.minute
        if current_minutes >= 18 * 60 + 30:
            raise HTTPException(400, "Booking for today is closed. Please select a future date.")
        if appointment.slot == "morning" and current_minutes >= 12 * 60 + 30:
            raise HTTPException(400, "Morning slot booking is closed for today.")

    doctor = await db.doctors.find_one({"doctor_id": appointment.doctor_id})
    if not doctor:
        raise HTTPException(404, "Doctor not found")

    doctor_obj_id = str(doctor["_id"])

    leave = await db.doctor_leaves.find_one({
        "doctor_id": doctor_obj_id,
        "start_date": {"$lte": appointment.date},
        "end_date": {"$gte": appointment.date},
        "approval_status": {"$in": ["approved", "auto_approved"]}
    })
    if leave:
        raise HTTPException(400, "Doctor is on leave for this date")

    availability = await db.doctor_availability.find_one({"doctor_id": doctor_obj_id, "date": appointment.date})
    if not availability:
        availability = {
            "is_available": True, "morning_slot_enabled": True,
            "afternoon_slot_enabled": True, "morning_capacity": 100, "afternoon_capacity": 100
        }
    elif not availability.get("is_available"):
        raise HTTPException(400, "Doctor is not available on this date")

    if appointment.slot == "morning":
        if not availability.get("morning_slot_enabled", True):
            raise HTTPException(400, "Morning slot not available")
        capacity = availability.get("morning_capacity", 100)
    else:
        if not availability.get("afternoon_slot_enabled", True):
            raise HTTPException(400, "Afternoon slot not available")
        capacity = availability.get("afternoon_capacity", 100)

    existing_bookings = await db.appointments.count_documents({
        "doctor_id": doctor_obj_id, "date": appointment.date,
        "slot": appointment.slot, "status": {"$in": ["confirmed", "pending"]}
    })
    if existing_bookings >= capacity:
        raise HTTPException(400, "Slot is full")

    duplicate = await db.appointments.find_one({
        "patient_id": patient_id, "doctor_id": doctor_obj_id,
        "date": appointment.date, "slot": appointment.slot,
        "status": {"$in": ["confirmed", "pending"]}
    })
    if duplicate:
        raise HTTPException(400, "You already have an appointment with this doctor at this time")

    token_number = existing_bookings + 1
    patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
    patient_name = patient.get("name", "Unknown") if patient else "Unknown"
    transaction_id = "TXN" + secrets.token_hex(4).upper()

    appointment_doc = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "doctor_id": doctor_obj_id,
        "doctor_name": doctor["name"],
        "date": appointment.date,
        "slot": appointment.slot,
        "slot_time": "9:30 AM - 12:30 PM" if appointment.slot == "morning" else "1:30 PM - 6:30 PM",
        "token_number": token_number,
        "symptoms": appointment.symptoms,
        "status": "confirmed",
        "transaction_id": transaction_id,
        "payment_method": appointment.payment_method or "card",
        "payment_amount": appointment.payment_amount or 318,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    result = await db.appointments.insert_one(appointment_doc)
    appointment_id = str(result.inserted_id)
    appointment_doc["_id"] = appointment_id

    await db.accounts.insert_one({
        "transaction_id": transaction_id,
        "appointment_id": appointment_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "doctor_id": doctor_obj_id,
        "doctor_name": doctor["name"],
        "appointment_date": appointment.date,
        "slot": appointment.slot,
        "consultation_fee": 299,
        "platform_fee": 19,
        "payment_amount": appointment.payment_amount or 318,
        "currency": "INR",
        "payment_method": appointment.payment_method or "card",
        "status": "success",
        "refund_amount": 0,
        "refund_at": None,
        "refund_transaction_id": None,
        "paid_at": datetime.utcnow(),
        "created_at": datetime.utcnow()
    })

    await db.patient_notifications.insert_one({
        "patient_id": patient_id,
        "type": "appointment_confirmed",
        "status": "confirmed",
        "message": (
            f"Your appointment with Dr. {doctor['name']} has been confirmed for "
            f"{appointment.date} ({appointment.slot} slot). Your token number is #{token_number}."
        ),
        "appointment_id": appointment_id,
        "token_number": token_number,
        "timestamp": datetime.utcnow(),
        "created_at": datetime.utcnow(),
        "is_read": False
    })

    await notify_doctor_new_appointment(db, appointment_id)

    return {
        "message": f"Appointment booked successfully! Your token number is #{token_number}",
        "appointment": serialize_doc(appointment_doc),
        "token_number": token_number
    }


@router.get("/appointments")
async def get_patient_appointments(
    current_user: dict = Depends(get_current_user),
    status: Optional[str] = None
):
    if current_user["role"] != "patient":
        raise HTTPException(403, "Patient access required")
    db = get_db()
    patient_id = current_user["user_id"]

    query = {"patient_id": patient_id}
    if status:
        query["status"] = status

    appointments = await db.appointments.find(query).sort([
        ("date", 1), ("slot", 1), ("created_at", 1)
    ]).to_list(None)

    enriched = []
    for apt in appointments:
        doctor = await db.doctors.find_one({"_id": ObjectId(apt["doctor_id"])})
        apt_data = serialize_doc(apt)
        apt_data["doctor"] = serialize_doc(doctor) if doctor else None
        enriched.append(apt_data)

    today = date.today().isoformat()
    upcoming = [a for a in enriched if a["status"] in ["confirmed", "pending"] and a["date"] >= today]
    completed = [a for a in enriched if a["status"] == "completed"]
    cancelled = [a for a in enriched if a["status"] == "cancelled"]

    return {
        "total": len(enriched),
        "upcoming": upcoming,
        "completed": completed,
        "cancelled": cancelled,
        "all": enriched
    }


@router.patch("/appointment/{appointment_id}/cancel")
async def cancel_appointment(
    appointment_id: str,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "patient":
        raise HTTPException(403, "Patient access required")
    db = get_db()
    patient_id = current_user["user_id"]

    result = await db.appointments.update_one(
        {
            "_id": ObjectId(appointment_id),
            "patient_id": patient_id,
            "status": {"$in": ["confirmed", "pending"]}
        },
        {"$set": {"status": "cancelled", "cancelled_at": datetime.utcnow(), "cancelled_by": "patient"}}
    )

    if result.matched_count == 0:
        raise HTTPException(404, "Appointment not found or already cancelled")

    await notify_appointment_status_change(db, appointment_id, "cancelled", "Cancelled by you")
    await notify_doctor_appointment_cancelled(db, appointment_id, "patient")

    refund_txn_id = "REF" + secrets.token_hex(4).upper()
    await db.accounts.update_one(
        {"appointment_id": appointment_id},
        {"$set": {"status": "refunded", "refund_amount": 318, "refund_at": datetime.utcnow(), "refund_transaction_id": refund_txn_id}}
    )

    await db.patient_notifications.insert_one({
        "patient_id": patient_id,
        "type": "refund_initiated",
        "status": "refund",
        "message": (
            f"Refund of ₹318 has been initiated for your cancelled appointment. "
            f"It will be credited to your original payment method within 2-3 business days. "
            f"Refund ID: {refund_txn_id}"
        ),
        "appointment_id": appointment_id,
        "refund_transaction_id": refund_txn_id,
        "timestamp": datetime.utcnow(),
        "created_at": datetime.utcnow(),
        "is_read": False
    })

    return {"message": "Appointment cancelled successfully"}