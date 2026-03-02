from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, date, timedelta
from typing import Optional
import calendar
import secrets
from bson import ObjectId

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.utils import serialize_doc, is_sunday_date, count_leaves_in_month
from app.models.schemas import DoctorAvailability, SlotToggle, LeaveRequest
from app.services.notifications import notify_appointment_status_change

router = APIRouter(prefix="/api/doctor", tags=["Doctor"])


# ── Availability ───────────────────────────────────────────────────────────────

@router.post("/availability/set")
async def set_doctor_availability(
    availability: DoctorAvailability,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]

    existing = await db.doctor_availability.find_one({"doctor_id": doctor_id, "date": availability.date})

    availability_doc = {
        "doctor_id": doctor_id,
        "date": availability.date,
        "is_available": availability.is_available,
        "morning_slot_enabled": availability.morning_slot_enabled,
        "afternoon_slot_enabled": availability.afternoon_slot_enabled,
        "morning_capacity": availability.morning_capacity,
        "afternoon_capacity": availability.afternoon_capacity,
        "morning_disabled_at": None,
        "afternoon_disabled_at": None,
        "morning_disable_counted": False,
        "afternoon_disable_counted": False,
        "notes": availability.notes,
        "updated_at": datetime.utcnow()
    }

    if existing:
        await db.doctor_availability.update_one({"_id": existing["_id"]}, {"$set": availability_doc})
        message = "Availability updated"
    else:
        availability_doc["created_at"] = datetime.utcnow()
        await db.doctor_availability.insert_one(availability_doc)
        message = "Availability set"

    return {"message": message, "availability": availability_doc}


@router.post("/slot/toggle")
async def toggle_slot(
    payload: SlotToggle,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]

    today_date = date.today()
    today = today_date.isoformat()

    if payload.date != today:
        raise HTTPException(400, "Slot toggling allowed only for today")

    availability = await db.doctor_availability.find_one({"doctor_id": doctor_id, "date": today})

    if not availability:
        availability_doc = {
            "doctor_id": doctor_id, "date": today,
            "is_available": True, "morning_slot_enabled": True, "afternoon_slot_enabled": True,
            "morning_capacity": 100, "afternoon_capacity": 100,
            "morning_disabled_at": None, "afternoon_disabled_at": None,
            "morning_disable_counted": False, "afternoon_disable_counted": False,
            "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()
        }
        result = await db.doctor_availability.insert_one(availability_doc)
        availability_doc["_id"] = result.inserted_id
        availability = availability_doc

    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    morning_disable_limit = 9 * 60 + 40
    afternoon_disable_limit = 13 * 60 + 40

    if payload.action == "disable":
        if not availability.get(f"{payload.slot}_slot_enabled", True):
            raise HTTPException(400, "Slot already disabled")

        other_slot = "afternoon" if payload.slot == "morning" else "morning"
        if not availability.get(f"{other_slot}_slot_enabled", True):
            raise HTTPException(400, "Only one slot can be disabled per day")

        if payload.slot == "morning" and current_minutes > morning_disable_limit:
            raise HTTPException(400, "Morning slot can only be disabled before 9:40 AM")
        if payload.slot == "afternoon" and current_minutes > afternoon_disable_limit:
            raise HTTPException(400, "Afternoon slot can only be disabled before 1:40 PM")

        first_day = date(now.year, now.month, 1).isoformat()
        last_day = date(now.year, now.month, calendar.monthrange(now.year, now.month)[1]).isoformat()

        disable_count = await db.doctor_availability.count_documents({
            "doctor_id": doctor_id,
            "date": {"$gte": first_day, "$lte": last_day},
            f"{payload.slot}_disable_counted": True
        })

        if disable_count >= 3:
            raise HTTPException(400, "Monthly disable limit reached (3 per slot)")

        update = {
            f"{payload.slot}_slot_enabled": False,
            f"{payload.slot}_disabled_at": now,
            f"{payload.slot}_disable_counted": True,
            "updated_at": datetime.utcnow()
        }

    else:  # enable
        if availability.get(f"{payload.slot}_slot_enabled", True):
            raise HTTPException(400, "Slot is already enabled")

        disabled_at = availability.get(f"{payload.slot}_disabled_at")
        if not disabled_at:
            raise HTTPException(400, "No disable timestamp found")

        diff = (now - disabled_at).total_seconds()
        if diff > 120:
            raise HTTPException(400, "Re-enable window expired (2 minutes)")

        update = {
            f"{payload.slot}_slot_enabled": True,
            f"{payload.slot}_disabled_at": None,
            "updated_at": datetime.utcnow()
        }

    await db.doctor_availability.update_one({"_id": availability["_id"]}, {"$set": update})
    return {"message": f"{payload.slot.capitalize()} slot {payload.action}d successfully"}


@router.get("/availability/today")
async def get_today_availability(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]
    today_date = date.today()
    today = today_date.isoformat()

    if is_sunday_date(today_date):
        return {
            "date": today, "is_available": False, "reason": "Sunday - Non working day",
            "morning_slot_enabled": False, "afternoon_slot_enabled": False,
            "morning_capacity": 100, "afternoon_capacity": 100,
            "morning_booked": 0, "afternoon_booked": 0
        }

    leave = await db.doctor_leaves.find_one({
        "doctor_id": doctor_id,
        "start_date": {"$lte": today}, "end_date": {"$gte": today},
        "approval_status": {"$in": ["approved", "auto_approved"]}
    })
    if leave:
        return {
            "date": today, "is_available": False, "reason": "On Leave",
            "morning_slot_enabled": False, "afternoon_slot_enabled": False,
            "morning_capacity": 100, "afternoon_capacity": 100,
            "morning_booked": 0, "afternoon_booked": 0
        }

    availability = await db.doctor_availability.find_one({"doctor_id": doctor_id, "date": today})

    morning_bookings = await db.appointments.count_documents({
        "doctor_id": doctor_id, "date": today, "slot": "morning",
        "status": {"$in": ["confirmed", "pending"]}
    })
    afternoon_bookings = await db.appointments.count_documents({
        "doctor_id": doctor_id, "date": today, "slot": "afternoon",
        "status": {"$in": ["confirmed", "pending"]}
    })

    if not availability:
        return {
            "date": today, "is_available": True,
            "morning_slot_enabled": True, "afternoon_slot_enabled": True,
            "morning_capacity": 100, "afternoon_capacity": 100,
            "morning_booked": morning_bookings, "afternoon_booked": afternoon_bookings,
            "morning_available": 100 - morning_bookings,
            "afternoon_available": 100 - afternoon_bookings
        }

    return {
        **serialize_doc(availability),
        "morning_booked": morning_bookings,
        "afternoon_booked": afternoon_bookings,
        "morning_available": availability["morning_capacity"] - morning_bookings,
        "afternoon_available": availability["afternoon_capacity"] - afternoon_bookings
    }


@router.get("/availability/range")
async def get_availability_range(
    start_date: str, end_date: str,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]

    availabilities = await db.doctor_availability.find({
        "doctor_id": doctor_id,
        "date": {"$gte": start_date, "$lte": end_date}
    }).to_list(None)

    leaves = await db.doctor_leaves.find({
        "doctor_id": doctor_id,
        "end_date": {"$gte": start_date}, "start_date": {"$lte": end_date},
        "status": "active"
    }).to_list(None)

    return {
        "availabilities": [serialize_doc(a) for a in availabilities],
        "leaves": [serialize_doc(l) for l in leaves]
    }


# ── Leave Management ───────────────────────────────────────────────────────────

@router.post("/leave/request")
async def request_leave(
    leave: LeaveRequest,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]

    try:
        start = datetime.strptime(leave.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(leave.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Invalid date format")

    if start > end:
        raise HTTPException(400, "Start date must be before end date")
    if start < date.today():
        raise HTTPException(400, "Cannot request leave for past dates")

    leave_days = 0
    d = start
    while d <= end:
        if not is_sunday_date(d):
            leave_days += 1
        d += timedelta(days=1)

    if leave_days == 0:
        raise HTTPException(400, "Selected dates contain only Sundays")

    MONTHLY_LIMIT = 3
    current_month_leaves = await count_leaves_in_month(db, doctor_id, start.year, start.month)
    needs_approval = (current_month_leaves + leave_days) > MONTHLY_LIMIT

    if needs_approval and not leave.reason:
        raise HTTPException(
            400,
            f"You already used {current_month_leaves} leave days this month. "
            f"Requesting {leave_days} more exceeds the {MONTHLY_LIMIT}-day limit. "
            "Reason is mandatory."
        )

    doctor = await db.doctors.find_one({"_id": ObjectId(doctor_id)})
    if not doctor:
        raise HTTPException(404, "Doctor not found")

    existing_appointments = await db.appointments.find({
        "doctor_id": doctor_id,
        "date": {"$gte": leave.start_date, "$lte": leave.end_date},
        "status": {"$in": ["confirmed", "pending"]}
    }).to_list(None)

    leave_doc = {
        "doctor_id": doctor_id,
        "doctor_name": doctor["name"],
        "created_by_admin_id": doctor.get("created_by_admin_id"),
        "start_date": leave.start_date,
        "end_date": leave.end_date,
        "leave_days": leave_days,
        "reason": leave.reason,
        "approval_status": "auto_approved" if not needs_approval else "pending",
        "needs_approval": needs_approval,
        "affected_appointments": len(existing_appointments),
        "admin_notes": "Auto-approved (within monthly limit)" if not needs_approval else None,
        "reviewed_at": datetime.utcnow() if not needs_approval else None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    result = await db.doctor_leaves.insert_one(leave_doc)

    if not needs_approval and existing_appointments:
        for apt in existing_appointments:
            apt_id = str(apt["_id"])

            # 1️⃣ Cancel appointment
            await db.appointments.update_one(
                {"_id": apt["_id"]},
                {"$set": {
                    "status": "cancelled",
                    "cancellation_reason": "Doctor on leave (auto-approved)",
                    "cancelled_by": "system",
                    "cancelled_at": datetime.utcnow()
                }}
            )

            # 2️⃣ Notify patient of cancellation
            await notify_appointment_status_change(
                db, apt_id, "cancelled", "Doctor is on leave"
            )

            # 3️⃣ Process refund in accounts
            refund_txn_id = "REF" + secrets.token_hex(4).upper()
            await db.accounts.update_one(
                {"appointment_id": apt_id},
                {"$set": {
                    "status": "refunded",
                    "refund_amount": 318,
                    "refund_at": datetime.utcnow(),
                    "refund_transaction_id": refund_txn_id,
                    "refund_reason": "Doctor leave auto-approved"
                }}
            )

            # 4️⃣ Send refund notification to patient
            await db.patient_notifications.insert_one({
                "patient_id": apt["patient_id"],
                "type": "refund_initiated",
                "status": "refund",
                "message": (
                    f"Refund of ₹318 has been initiated for your cancelled appointment "
                    f"(Doctor is on leave). It will be credited to your original "
                    f"payment method within 2-3 business days. Refund ID: {refund_txn_id}"
                ),
                "appointment_id": apt_id,
                "refund_transaction_id": refund_txn_id,
                "timestamp": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                "is_read": False
            })

    return {
        "message": "Leave auto-approved" if not needs_approval else "Leave request sent for admin approval",
        "leave": serialize_doc(leave_doc),
        "needs_approval": needs_approval,
        "current_month_leaves": current_month_leaves,
        "monthly_limit": MONTHLY_LIMIT,
        "cancelled_appointments": 0 if needs_approval else len(existing_appointments)
    }


@router.get("/leave/list")
async def list_leaves(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    leaves = await db.doctor_leaves.find(
        {"doctor_id": current_user["user_id"]}
    ).sort("start_date", -1).to_list(None)
    return {"leaves": [serialize_doc(l) for l in leaves]}


@router.delete("/leave/{leave_id}")
async def cancel_leave(leave_id: str, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]

    leave = await db.doctor_leaves.find_one({"_id": ObjectId(leave_id), "doctor_id": doctor_id})
    if not leave:
        raise HTTPException(404, "Leave not found")
    if leave["approval_status"] in ["rejected", "cancelled"]:
        raise HTTPException(400, f"Cannot cancel a {leave['approval_status']} leave")

    if leave["approval_status"] in ["approved", "auto_approved"]:
        await db.appointments.update_many(
            {
                "doctor_id": doctor_id,
                "date": {"$gte": leave["start_date"], "$lte": leave["end_date"]},
                "status": "cancelled",
                "cancellation_reason": {"$in": [
                    "Doctor on leave (auto-approved)",
                    "Doctor leave approved by admin"
                ]}
            },
            {
                "$set": {"status": "confirmed", "updated_at": datetime.utcnow()},
                "$unset": {"cancelled_at": "", "cancellation_reason": ""}
            }
        )

    await db.doctor_leaves.update_one(
        {"_id": ObjectId(leave_id)},
        {"$set": {"approval_status": "cancelled", "updated_at": datetime.utcnow()}}
    )
    return {"message": "Leave cancelled successfully"}


# ── Appointments ───────────────────────────────────────────────────────────────

@router.get("/appointments/today")
async def get_today_appointments(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]
    today = date.today().isoformat()

    appointments = await db.appointments.find({
        "doctor_id": doctor_id, "date": today,
        "status": {"$in": ["confirmed", "pending"]}
    }).to_list(None)

    enriched = []
    for apt in appointments:
        patient = await db.patients.find_one({"_id": ObjectId(apt["patient_id"])})
        apt_data = serialize_doc(apt)
        apt_data["patient"] = serialize_doc(patient) if patient else None
        enriched.append(apt_data)

    morning = [a for a in enriched if a["slot"] == "morning"]
    afternoon = [a for a in enriched if a["slot"] == "afternoon"]

    return {
        "date": today, "total": len(appointments),
        "morning": {"count": len(morning), "appointments": morning},
        "afternoon": {"count": len(afternoon), "appointments": afternoon}
    }


@router.get("/appointments/future")
async def get_future_appointments(
    current_user: dict = Depends(get_current_user), limit: int = 50
):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    appointments = await db.appointments.find({
        "doctor_id": doctor_id,
        "date": {"$gte": tomorrow},
        "status": {"$in": ["confirmed", "pending", "completed"]}
    }).sort("date", 1).limit(limit).to_list(None)

    enriched = []
    for apt in appointments:
        patient = await db.patients.find_one({"_id": ObjectId(apt["patient_id"])})
        apt_data = serialize_doc(apt)
        apt_data["patient"] = serialize_doc(patient) if patient else None
        enriched.append(apt_data)

    by_date = {}
    for apt in enriched:
        apt_date = apt["date"]
        if apt_date not in by_date:
            by_date[apt_date] = {"date": apt_date, "total": 0, "morning": [], "afternoon": []}
        by_date[apt_date]["total"] += 1
        by_date[apt_date]["morning" if apt["slot"] == "morning" else "afternoon"].append(apt)

    return {"total": len(enriched), "appointments": enriched, "by_date": list(by_date.values())}


@router.get("/appointments/upcoming")
async def get_upcoming_appointments(
    current_user: dict = Depends(get_current_user), limit: int = 50
):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]
    today = date.today().isoformat()

    appointments = await db.appointments.find({
        "doctor_id": doctor_id, "date": {"$gte": today},
        "status": {"$in": ["confirmed", "pending"]}
    }).sort("date", 1).limit(limit).to_list(None)

    enriched = []
    for apt in appointments:
        patient = await db.patients.find_one({"_id": ObjectId(apt["patient_id"])})
        apt_data = serialize_doc(apt)
        apt_data["patient"] = serialize_doc(patient) if patient else None
        enriched.append(apt_data)

    return {"total": len(enriched), "appointments": enriched}


@router.patch("/appointment/{appointment_id}/complete")
async def mark_appointment_completed(
    appointment_id: str,
    notes: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]

    appointment = await db.appointments.find_one({
        "_id": ObjectId(appointment_id), "doctor_id": doctor_id
    })
    if not appointment:
        raise HTTPException(404, "Appointment not found")
    if appointment["status"] == "completed":
        raise HTTPException(400, "Appointment already completed")

    await db.appointments.update_one(
        {"_id": ObjectId(appointment_id)},
        {"$set": {
            "status": "completed",
            "consultation_notes": notes,
            "completed_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }}
    )

    await notify_appointment_status_change(db, appointment_id, "completed")
    return {"message": "Consultation marked as completed"}


# ── Notifications ──────────────────────────────────────────────────────────────

@router.get("/notifications")
async def get_doctor_notifications(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "doctor":
        raise HTTPException(403, "Doctor access required")
    db = get_db()
    doctor_id = current_user["user_id"]

    leaves = await db.doctor_leaves.find({
        "doctor_id": doctor_id,
        "approval_status": {"$in": ["approved", "rejected", "auto_approved"]},
        "reviewed_at": {"$exists": True}
    }).sort("reviewed_at", -1).to_list(None)

    notifications = []

    for leave in leaves:
        if leave["approval_status"] == "auto_approved":
            message = f"Your leave from {leave['start_date']} to {leave['end_date']} was auto-approved (within monthly limit)"
        elif leave["approval_status"] == "approved":
            message = f"Your leave from {leave['start_date']} to {leave['end_date']} was approved by admin"
        else:
            message = f"Your leave from {leave['start_date']} to {leave['end_date']} was rejected"

        notifications.append({
            "_id": str(leave["_id"]),
            "type": "leave_decision",
            "status": leave["approval_status"],
            "message": message,
            "admin_notes": leave.get("admin_notes"),
            "timestamp": leave["reviewed_at"]
        })

    appointment_notifications = await db.doctor_notifications.find(
        {"doctor_id": doctor_id}
    ).sort("created_at", -1).limit(50).to_list(None)

    for notif in appointment_notifications:
        notifications.append({
            "_id": str(notif["_id"]),
            "type": notif.get("type", "appointment"),
            "status": notif.get("status", "info"),
            "message": notif["message"],
            "appointment_id": notif.get("appointment_id"),
            "appointment_date": notif.get("appointment_date"),
            "slot": notif.get("slot"),
            "patient_name": notif.get("patient_name"),
            "token_number": notif.get("token_number"),
            "timestamp": notif.get("timestamp", notif.get("created_at"))
        })

    notifications.sort(key=lambda x: x["timestamp"], reverse=True)

    return {"notifications": notifications[:50], "unread_count": len(notifications)}