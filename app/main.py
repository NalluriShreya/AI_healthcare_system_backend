from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId

from core.database import connect_db, close_db, get_db
from core.security import get_current_user
from core.utils import serialize_doc
from routers import auth, otp, patient, doctor, admin, prediction

app = FastAPI(title="AI Healthcare System API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database lifecycle
app.add_event_handler("startup", connect_db)
app.add_event_handler("shutdown", close_db)

# Register routers
app.include_router(auth.router, prefix="/api", tags=["Auth"])
app.include_router(otp.router, prefix="/api", tags=["OTP"])
app.include_router(patient.router)
app.include_router(doctor.router)
app.include_router(admin.router)
app.include_router(prediction.router)


# ── Misc Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "AI Healthcare System API", "status": "running"}


@app.get("/api/departments")
async def get_departments():
    db = get_db()
    departments = await db.doctors.distinct("department", {"status": "active"})
    return {"departments": sorted(departments)}


@app.get("/api/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    db = get_db()
    role = current_user["role"]
    user_id = ObjectId(current_user["user_id"])

    if role == "patient":
        user = await db.patients.find_one({"_id": user_id})
    elif role == "doctor":
        user = await db.doctors.find_one({"_id": user_id})
    elif role == "admin":
        user = await db.admins.find_one({"_id": user_id})
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid role")

    if not user:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="User not found")

    return serialize_doc(user)