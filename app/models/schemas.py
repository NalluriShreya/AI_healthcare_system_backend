from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal
from bson import ObjectId


# ── Auth / User Models ─────────────────────────────────────────────────────────

class PatientSignup(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: str = Field(..., pattern=r"^\+?[1-9]\d{9,14}$")
    password: str = Field(..., min_length=8)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1234567890",
                "password": "SecurePass123!"
            }
        }


class PatientLogin(BaseModel):
    email: EmailStr
    password: str


class DoctorSignup(BaseModel):
    email: EmailStr
    phone: str = Field(..., pattern=r"^\+?[1-9]\d{9,14}$")
    password: str = Field(..., min_length=8)

    class Config:
        json_schema_extra = {
            "example": {
                "email": "doctor@hospital.com",
                "phone": "+1234567890",
                "password": "SecurePass123!"
            }
        }


class DoctorLogin(BaseModel):
    email: EmailStr
    password: str


class AdminLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


# ── OTP Models ─────────────────────────────────────────────────────────────────

class OTPRequest(BaseModel):
    phone: str = Field(..., pattern=r"^\+?[1-9]\d{9,14}$")
    role: Literal["patient", "doctor"]


class OTPVerify(BaseModel):
    phone: str = Field(..., pattern=r"^\+?[1-9]\d{9,14}$")
    role: Literal["patient", "doctor"]
    otp_code: str = Field(..., min_length=6, max_length=6)


# ── Forgot Password Models ─────────────────────────────────────────────────────

class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    role: Literal["patient", "doctor", "admin"]


class VerifyResetOTPRequest(BaseModel):
    email: EmailStr
    role: Literal["patient", "doctor", "admin"]
    otp_code: str = Field(..., min_length=6, max_length=6)


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    role: Literal["patient", "doctor", "admin"]
    otp_code: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=8)


# ── Admin Models ───────────────────────────────────────────────────────────────

class DoctorCreate(BaseModel):
    doctor_id: str = Field(..., min_length=1, max_length=50)
    name: str
    email: EmailStr
    phone_number: str = Field(..., pattern=r"^\+?[1-9]\d{9,14}$")
    department: str
    specialization: str
    qualification: str
    status: Literal["pending", "active", "inactive"] = "pending"


class DoctorUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = Field(None, pattern=r"^\+?[1-9]\d{9,14}$")
    department: Optional[str] = None
    specialization: Optional[str] = None
    qualification: Optional[str] = None
    status: Optional[Literal["pending", "active", "inactive"]] = None


class DoctorStatusUpdate(BaseModel):
    status: Literal["active", "inactive"]


class PatientStatusUpdate(BaseModel):
    is_active: bool


# ── Availability & Appointment Models ─────────────────────────────────────────

class DoctorAvailability(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    is_available: bool = True
    morning_slot_enabled: bool = True
    afternoon_slot_enabled: bool = True
    morning_capacity: int = 100
    afternoon_capacity: int = 100
    notes: Optional[str] = None


class SlotToggle(BaseModel):
    date: str
    slot: Literal["morning", "afternoon"]
    action: Literal["disable", "enable"]


class LeaveRequest(BaseModel):
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    reason: Optional[str] = None


class LeaveApproval(BaseModel):
    leave_id: str
    action: Literal["approve", "reject"]
    admin_notes: Optional[str] = None


class AppointmentCreate(BaseModel):
    doctor_id: str
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    slot: Literal["morning", "afternoon"]
    symptoms: Optional[str] = None
    payment_method: Optional[str] = "card"
    payment_amount: Optional[int] = 318


class AppointmentUpdate(BaseModel):
    status: Literal["confirmed", "cancelled", "completed"]
    notes: Optional[str] = None


# ── Patient Profile ────────────────────────────────────────────────────────────

class PatientProfileUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, pattern=r"^\+?[1-9]\d{9,14}$")
    current_password: Optional[str] = None
    new_password: Optional[str] = Field(None, min_length=8)


# ── Mongo ObjectId helper ──────────────────────────────────────────────────────

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)