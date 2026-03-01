from datetime import datetime, date
from bson import ObjectId
from typing import Optional


async def notify_appointment_status_change(db, appointment_id: str, new_status: str, reason: Optional[str] = None):
    """Send notification to patient when appointment status changes"""
    appointment = await db.appointments.find_one({"_id": ObjectId(appointment_id)})
    if not appointment:
        return

    patient_id = appointment["patient_id"]
    doctor_name = appointment.get("doctor_name", "Doctor")
    apt_date = appointment["date"]
    slot = appointment["slot"]
    token_number = appointment.get("token_number", "N/A")

    if new_status == "confirmed":
        message = (
            f"Your appointment with Dr. {doctor_name} has been CONFIRMED for "
            f"{apt_date} ({slot} slot). Token: #{token_number}"
        )
    elif new_status == "cancelled":
        message = (
            f"Your appointment with Dr. {doctor_name} on {apt_date} ({slot} slot) "
            f"has been CANCELLED."
        )
        if reason:
            message += f" Reason: {reason}"
    elif new_status == "completed":
        message = (
            f"Your appointment with Dr. {doctor_name} on {apt_date} ({slot} slot) "
            f"has been marked as COMPLETED."
        )
    else:
        message = f"Appointment status updated to {new_status}"

    await db.patient_notifications.insert_one({
        "patient_id": patient_id,
        "type": "appointment_status_change",
        "status": new_status,
        "message": message,
        "appointment_id": str(appointment_id),
        "token_number": token_number,
        "timestamp": datetime.utcnow(),
        "created_at": datetime.utcnow(),
        "is_read": False
    })


async def notify_doctor_new_appointment(db, appointment_id: str):
    """Send notification to doctor when a new appointment is booked"""
    appointment = await db.appointments.find_one({"_id": ObjectId(appointment_id)})
    if not appointment:
        return

    doctor_id = appointment["doctor_id"]
    patient_id = appointment["patient_id"]
    apt_date = appointment["date"]
    slot = appointment["slot"]
    token_number = appointment.get("token_number", "N/A")

    patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
    patient_name = patient.get("name", "A patient") if patient else "A patient"

    try:
        appointment_date = datetime.strptime(apt_date, "%Y-%m-%d").date()
        today = date.today()

        if appointment_date == today:
            message = f"New appointment booked for TODAY ({slot} slot): {patient_name} - Token #{token_number}"
            notification_type = "appointment_booked_today"
        elif appointment_date > today:
            date_str = appointment_date.strftime('%B %d, %Y')
            message = f"New appointment booked for {date_str} ({slot} slot): {patient_name} - Token #{token_number}"
            notification_type = "appointment_booked_future"
        else:
            return
    except ValueError:
        return

    await db.doctor_notifications.insert_one({
        "doctor_id": doctor_id,
        "type": notification_type,
        "status": "approved",
        "message": message,
        "appointment_id": str(appointment_id),
        "appointment_date": apt_date,
        "slot": slot,
        "patient_name": patient_name,
        "token_number": token_number,
        "timestamp": datetime.utcnow(),
        "created_at": datetime.utcnow(),
        "is_read": False
    })


async def notify_doctor_appointment_cancelled(db, appointment_id: str, cancelled_by: str):
    """Send notification to doctor when an appointment is cancelled"""
    appointment = await db.appointments.find_one({"_id": ObjectId(appointment_id)})
    if not appointment:
        return

    doctor_id = appointment["doctor_id"]
    patient_id = appointment["patient_id"]
    apt_date = appointment["date"]
    slot = appointment["slot"]
    token_number = appointment.get("token_number", "N/A")

    patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
    patient_name = patient.get("name", "A patient") if patient else "A patient"

    try:
        appointment_date = datetime.strptime(apt_date, "%Y-%m-%d").date()
        today = date.today()

        if appointment_date == today:
            message = f"Appointment CANCELLED for TODAY ({slot} slot): {patient_name} - Token #{token_number}"
        elif appointment_date > today:
            date_str = appointment_date.strftime('%B %d, %Y')
            message = f"Appointment CANCELLED for {date_str} ({slot} slot): {patient_name} - Token #{token_number}"
        else:
            date_str = appointment_date.strftime('%B %d, %Y')
            message = f"Appointment CANCELLED for {date_str} ({slot} slot): {patient_name} - Token #{token_number}"
    except ValueError:
        message = f"Appointment CANCELLED: {patient_name} - Token #{token_number}"

    await db.doctor_notifications.insert_one({
        "doctor_id": doctor_id,
        "type": "appointment_cancelled",
        "status": "rejected",
        "message": message,
        "appointment_id": str(appointment_id),
        "appointment_date": apt_date,
        "slot": slot,
        "patient_name": patient_name,
        "token_number": token_number,
        "cancelled_by": cancelled_by,
        "timestamp": datetime.utcnow(),
        "created_at": datetime.utcnow(),
        "is_read": False
    })


async def cancel_appointments_due_to_leave(db, leave):
    import secrets
    doctor_id = leave["doctor_id"]
    start_date = leave["start_date"]
    end_date = leave["end_date"]

    appointments = await db.appointments.find({
        "doctor_id": doctor_id,
        "date": {"$gte": start_date, "$lte": end_date},
        "status": {"$in": ["confirmed", "pending"]}
    }).to_list(None)

    for apt in appointments:
        await db.appointments.update_one(
            {"_id": apt["_id"]},
            {
                "$set": {
                    "status": "cancelled",
                    "cancelled_reason": "Doctor is on leave",
                    "cancelled_by": "system",
                    "updated_at": datetime.utcnow()
                }
            }
        )

        await notify_appointment_status_change(db, str(apt["_id"]), "cancelled", "Doctor is on leave")

        refund_txn_id = "REF" + secrets.token_hex(4).upper()
        await db.accounts.update_one(
            {"appointment_id": str(apt["_id"])},
            {
                "$set": {
                    "status": "refunded",
                    "refund_amount": 318,
                    "refund_at": datetime.utcnow(),
                    "refund_transaction_id": refund_txn_id,
                    "refund_reason": "Doctor leave approved by admin"
                }
            }
        )

        await db.patient_notifications.insert_one({
            "patient_id": apt["patient_id"],
            "type": "refund_initiated",
            "status": "refund",
            "message": (
                f"Refund of ₹318 has been initiated for your cancelled appointment "
                f"(Doctor is on leave). It will be credited to your original "
                f"payment method within 2-3 business days. Refund ID: {refund_txn_id}"
            ),
            "appointment_id": str(apt["_id"]),
            "refund_transaction_id": refund_txn_id,
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow(),
            "is_read": False
        })