from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, date, timedelta
from typing import Optional, Literal
import calendar
from bson import ObjectId
import secrets

from core.database import get_db
from core.security import get_current_user
from core.utils import serialize_doc
from models.schemas import (
    DoctorCreate, DoctorUpdate, DoctorStatusUpdate,
    PatientStatusUpdate, LeaveApproval
)
from services.notifications import notify_appointment_status_change

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ── Doctor Management ──────────────────────────────────────────────────────────

@router.post("/doctor/create")
async def create_doctor(
    doctor: DoctorCreate,
    current_user: dict = Depends(get_current_user)
):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()

    admin = await db.admins.find_one({"_id": ObjectId(current_user["user_id"])})
    if not admin:
        raise HTTPException(404, "Admin not found")

    if await db.doctors.find_one({"doctor_id": doctor.doctor_id}):
        raise HTTPException(400, "Doctor ID already exists")
    if await db.doctors.find_one({"email": doctor.email}):
        raise HTTPException(400, "Email already registered")
    if await db.doctors.find_one({"phone": doctor.phone_number}):
        raise HTTPException(400, "Phone number already registered")
    if await db.patients.find_one({"email": doctor.email}):
        raise HTTPException(400, "Email already registered to a patient")
    if await db.admins.find_one({"email": doctor.email}):
        raise HTTPException(400, "Email already registered to an admin")
    if await db.patients.find_one({"phone": doctor.phone_number}):
        raise HTTPException(400, "Phone number already registered to a patient")
    if await db.admins.find_one({"phone": doctor.phone_number}):
        raise HTTPException(400, "Phone number already registered to an admin")

    doctor_doc = {
        "doctor_id": doctor.doctor_id,
        "name": doctor.name,
        "email": doctor.email,
        "phone": doctor.phone_number,
        "department": doctor.department,
        "specialization": doctor.specialization,
        "qualification": doctor.qualification,
        "password_hash": None,
        "status": doctor.status,
        "role": "doctor",
        "created_by_admin_id": current_user["user_id"],
        "created_by_admin_name": admin["name"],
        "created_at": datetime.utcnow(),
        "last_login": None,
    }

    result = await db.doctors.insert_one(doctor_doc)
    doctor_doc["_id"] = str(result.inserted_id)
    return {"message": "Doctor created", "doctor": serialize_doc(doctor_doc)}


@router.patch("/doctor/{doctor_id}/update")
async def update_doctor(
    doctor_id: str,
    doctor_update: DoctorUpdate,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()

    doctor = await db.doctors.find_one({"doctor_id": doctor_id})
    if not doctor:
        raise HTTPException(404, "Doctor not found")

    update_data = {}

    if doctor_update.name is not None:
        update_data["name"] = doctor_update.name
    if doctor_update.email is not None:
        existing = await db.doctors.find_one({"email": doctor_update.email, "doctor_id": {"$ne": doctor_id}})
        if existing:
            raise HTTPException(400, "Email already in use")
        update_data["email"] = doctor_update.email
    if doctor_update.phone_number is not None:
        existing = await db.doctors.find_one({"phone": doctor_update.phone_number, "doctor_id": {"$ne": doctor_id}})
        if existing:
            raise HTTPException(400, "Phone already in use")
        update_data["phone"] = doctor_update.phone_number
    if doctor_update.department is not None:
        update_data["department"] = doctor_update.department
    if doctor_update.specialization is not None:
        update_data["specialization"] = doctor_update.specialization
    if doctor_update.qualification is not None:
        update_data["qualification"] = doctor_update.qualification
    if doctor_update.status is not None:
        update_data["status"] = doctor_update.status

    if not update_data:
        raise HTTPException(400, "No fields to update")

    update_data["updated_at"] = datetime.utcnow()
    await db.doctors.update_one({"doctor_id": doctor_id}, {"$set": update_data})

    updated_doctor = await db.doctors.find_one({"doctor_id": doctor_id})
    return {"message": "Doctor updated successfully", "doctor": serialize_doc(updated_doctor)}


@router.get("/doctors")
async def get_all_doctors(
    current_user: dict = Depends(get_current_user),
    status_filter: Optional[str] = None,
    department: Optional[str] = None
):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()

    query = {}
    if status_filter and status_filter != "all":
        query["status"] = status_filter
    if department and department != "all":
        query["department"] = department

    doctors = await db.doctors.find(query).sort("created_at", -1).to_list(None)
    return {"total": len(doctors), "doctors": [serialize_doc(d) for d in doctors]}


@router.get("/doctor/{doctor_id}")
async def get_doctor_details(doctor_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()
    doctor = await db.doctors.find_one({"doctor_id": doctor_id})
    if not doctor:
        raise HTTPException(404, "Doctor not found")
    return {"doctor": serialize_doc(doctor)}


@router.patch("/doctor/{doctor_id}/status")
async def update_doctor_status(
    doctor_id: str,
    payload: DoctorStatusUpdate,
    current_user: dict = Depends(get_current_user)
):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()
    result = await db.doctors.update_one({"doctor_id": doctor_id}, {"$set": {"status": payload.status}})
    if result.matched_count == 0:
        raise HTTPException(404, "Doctor not found")
    return {"message": "Doctor status updated"}


# ── Patient Management ─────────────────────────────────────────────────────────

@router.get("/patients")
async def get_all_patients(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()
    patients = await db.patients.find().sort("created_at", -1).to_list(None)
    return {"total": len(patients), "patients": [serialize_doc(p) for p in patients]}


@router.patch("/patient/{patient_id}/status")
async def update_patient_status(
    patient_id: str,
    payload: PatientStatusUpdate,
    current_user: dict = Depends(get_current_user)
):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()
    result = await db.patients.update_one(
        {"_id": ObjectId(patient_id)}, {"$set": {"is_active": payload.is_active}}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Patient not found")
    return {"message": "Patient status updated"}


# ── Leave Management ───────────────────────────────────────────────────────────

@router.get("/leave-requests")
async def get_leave_requests(
    current_user: dict = Depends(get_current_user),
    status_filter: Optional[str] = "all"
):
    if current_user["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()
    admin_id = current_user["user_id"]

    query = {"created_by_admin_id": admin_id}
    if status_filter == "pending":
        query["approval_status"] = "pending"
    elif status_filter == "approved":
        query["approval_status"] = {"$in": ["approved", "auto_approved"]}
    elif status_filter == "rejected":
        query["approval_status"] = "rejected"
    elif status_filter != "all":
        raise HTTPException(400, "Invalid status_filter value")

    leaves = await db.doctor_leaves.find(query).sort("created_at", -1).to_list(None)
    serialized = [serialize_doc(l) for l in leaves]

    return {
        "total": len(serialized),
        "pending": [l for l in serialized if l["approval_status"] == "pending"],
        "approved": [l for l in serialized if l["approval_status"] in ["approved", "auto_approved"]],
        "rejected": [l for l in serialized if l["approval_status"] == "rejected"],
        "all": serialized
    }


@router.post("/leave/review")
async def review_leave_request(
    approval: LeaveApproval,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()
    admin_id = current_user["user_id"]

    leave = await db.doctor_leaves.find_one({
        "_id": ObjectId(approval.leave_id), "created_by_admin_id": admin_id
    })
    if not leave:
        raise HTTPException(404, "Leave request not found")
    if leave["approval_status"] != "pending":
        raise HTTPException(400, f"Leave is already {leave['approval_status']}")

    new_status = "approved" if approval.action == "approve" else "rejected"

    await db.doctor_leaves.update_one(
        {"_id": ObjectId(approval.leave_id)},
        {"$set": {
            "approval_status": new_status,
            "admin_notes": approval.admin_notes,
            "reviewed_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }}
    )

    if approval.action == "approve":
        existing_appointments = await db.appointments.find({
            "doctor_id": leave["doctor_id"],
            "date": {"$gte": leave["start_date"], "$lte": leave["end_date"]},
            "status": {"$in": ["confirmed", "pending"]}
        }).to_list(None)

        for apt in existing_appointments:
            await db.appointments.update_one(
                {"_id": apt["_id"]},
                {"$set": {
                    "status": "cancelled",
                    "cancellation_reason": "Doctor leave approved by admin",
                    "cancelled_by": "admin",
                    "cancelled_at": datetime.utcnow()
                }}
            )
            await notify_appointment_status_change(db, str(apt["_id"]), "cancelled", "Doctor leave approved by admin")

            refund_txn_id = "REF" + secrets.token_hex(4).upper()
            await db.accounts.update_one(
                {"appointment_id": str(apt["_id"])},
                {"$set": {
                    "status": "refunded",
                    "refund_amount": 318,
                    "refund_at": datetime.utcnow(),
                    "refund_transaction_id": refund_txn_id,
                    "refund_reason": "Doctor leave approved by admin"
                }}
            )

            await db.patient_notifications.insert_one({
                "patient_id": apt["patient_id"],
                "type": "refund_initiated",
                "status": "refund",
                "message": (
                    f"Refund of ₹318 has been initiated for your cancelled appointment "
                    f"(Doctor on approved leave). It will be credited to your original "
                    f"payment method within 2-3 business days. Refund ID: {refund_txn_id}"
                ),
                "appointment_id": str(apt["_id"]),
                "refund_transaction_id": refund_txn_id,
                "timestamp": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                "is_read": False
            })

        return {"message": "Leave approved", "cancelled_appointments": len(existing_appointments)}

    return {"message": "Leave rejected"}


# ── Analytics ──────────────────────────────────────────────────────────────────

@router.get("/analytics")
async def get_system_analytics(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    db = get_db()
    admin_id = current_user["user_id"]
    today = date.today().isoformat()
    now = datetime.utcnow()

    first_day_month = date(now.year, now.month, 1).isoformat()
    last_day_month = date(now.year, now.month, calendar.monthrange(now.year, now.month)[1]).isoformat()

    total_doctors = await db.doctors.count_documents({"created_by_admin_id": admin_id})
    active_doctors = await db.doctors.count_documents({"created_by_admin_id": admin_id, "status": "active"})
    inactive_doctors = await db.doctors.count_documents({"created_by_admin_id": admin_id, "status": "inactive"})
    pending_doctors = await db.doctors.count_documents({"created_by_admin_id": admin_id, "status": "pending"})

    total_patients = await db.patients.count_documents({})
    active_patients = await db.patients.count_documents({"is_active": True})
    new_patients_month = await db.patients.count_documents({
        "created_at": {"$gte": datetime(now.year, now.month, 1)}
    })

    total_appointments = await db.appointments.count_documents({})
    today_appointments = await db.appointments.count_documents({"date": today})
    confirmed_appointments = await db.appointments.count_documents({"status": "confirmed"})
    completed_appointments = await db.appointments.count_documents({"status": "completed"})
    cancelled_appointments = await db.appointments.count_documents({"status": "cancelled"})
    month_appointments = await db.appointments.count_documents({
        "date": {"$gte": first_day_month, "$lte": last_day_month}
    })

    accounts = await db.accounts.find({"status": "success"}).to_list(None)
    total_revenue = sum(a.get("payment_amount", 0) for a in accounts)

    month_accounts = await db.accounts.find({
        "status": "success",
        "paid_at": {"$gte": datetime(now.year, now.month, 1)}
    }).to_list(None)
    month_revenue = sum(a.get("payment_amount", 0) for a in month_accounts)

    refunded_accounts = await db.accounts.find({"status": "refunded"}).to_list(None)
    total_refunded = sum(a.get("refund_amount", 0) for a in refunded_accounts)

    daily_stats = []
    for i in range(6, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        count = await db.appointments.count_documents({"date": d})
        completed = await db.appointments.count_documents({"date": d, "status": "completed"})
        cancelled = await db.appointments.count_documents({"date": d, "status": "cancelled"})
        revenue_docs = await db.accounts.find({"status": "success", "appointment_date": d}).to_list(None)
        revenue = sum(a.get("payment_amount", 0) for a in revenue_docs)
        daily_stats.append({
            "date": d,
            "label": datetime.strptime(d, "%Y-%m-%d").strftime("%a"),
            "total": count, "completed": completed, "cancelled": cancelled, "revenue": revenue
        })

    doctor_list = await db.doctors.find({"created_by_admin_id": admin_id, "status": "active"}).to_list(None)
    dept_map = {}
    for doc in doctor_list:
        dept = doc.get("department", "Unknown")
        dept_map[dept] = dept_map.get(dept, 0) + 1
    departments = [{"name": k, "count": v} for k, v in sorted(dept_map.items(), key=lambda x: -x[1])]

    pipeline = [
        {"$group": {"_id": "$doctor_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}, {"$limit": 5}
    ]
    top_doctor_apts = await db.appointments.aggregate(pipeline).to_list(None)
    top_doctors = []
    for item in top_doctor_apts:
        doc = await db.doctors.find_one({"_id": ObjectId(item["_id"])})
        if doc:
            top_doctors.append({
                "name": doc.get("name"), "department": doc.get("department"),
                "specialization": doc.get("specialization"), "appointments": item["count"]
            })

    pending_leaves = await db.doctor_leaves.count_documents(
        {"created_by_admin_id": admin_id, "approval_status": "pending"}
    )
    approved_leaves = await db.doctor_leaves.count_documents(
        {"created_by_admin_id": admin_id, "approval_status": {"$in": ["approved", "auto_approved"]}}
    )

    return {
        "doctors": {"total": total_doctors, "active": active_doctors, "inactive": inactive_doctors, "pending": pending_doctors},
        "patients": {"total": total_patients, "active": active_patients, "new_this_month": new_patients_month},
        "appointments": {
            "total": total_appointments, "today": today_appointments,
            "this_month": month_appointments, "confirmed": confirmed_appointments,
            "completed": completed_appointments, "cancelled": cancelled_appointments,
            "completion_rate": round((completed_appointments / total_appointments * 100), 1) if total_appointments > 0 else 0
        },
        "revenue": {"total": total_revenue, "this_month": month_revenue, "total_refunded": total_refunded, "net": total_revenue - total_refunded},
        "daily_stats": daily_stats,
        "departments": departments,
        "top_doctors": top_doctors,
        "leaves": {"pending": pending_leaves, "approved": approved_leaves}
    }