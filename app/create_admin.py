import asyncio
import os
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = "healthcare_db"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

async def create_admin():
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    email = "shr_admin@gmail.com"

    existing_admin = await db.admins.find_one({"email": email})
    if existing_admin:
        print("❌ Admin already exists")
        return

    admin_doc = {
        "name": "Shreya",
        "email": email,
        "password_hash": hash_password("Admin@123"),
        "role": "admin",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow()
    }

    await db.admins.insert_one(admin_doc)
    print("✅ Admin account created successfully")

    client.close()

asyncio.run(create_admin())
