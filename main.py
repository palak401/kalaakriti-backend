import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv
import hashlib

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignupData(BaseModel):
    name: str
    mobile: str
    email: str
    business_name: str
    region: str
    password: str


class LoginData(BaseModel):
    identifier: str
    password: str


class ProductData(BaseModel):
    artisan_id: str
    title: str
    description: str = ""
    category: str = ""
    price: float = 0


class PricingData(BaseModel):
    raw_material_cost: float
    labour_cost: float
    packaging_cost: float
    category: str = ""


CATEGORY_MARGINS = {
    "textiles": 0.35,
    "pottery": 0.40,
    "jewelry": 0.50,
    "woodwork": 0.35,
    "leather": 0.35,
    "paintings": 0.45,
    "handicrafts": 0.35,
}
DEFAULT_MARGIN = 0.30


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


@app.get("/")
def read_root():
    return {"message": "Kalaakriti backend is running!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/signup")
def signup(data: SignupData):
    existing_email = supabase.table("artisans").select("id").eq("email", data.email).execute()
    existing_mobile = supabase.table("artisans").select("id").eq("mobile", data.mobile).execute()

    problems = []
    if existing_email.data:
        problems.append("email")
    if existing_mobile.data:
        problems.append("mobile")

    if problems:
        if "email" in problems and "mobile" in problems:
            detail = "An account with this email and mobile number already exists."
        elif "email" in problems:
            detail = "An account with this email already exists."
        else:
            detail = "An account with this mobile number already exists."
        raise HTTPException(status_code=400, detail=detail)

    try:
        result = supabase.table("artisans").insert({
            "name": data.name,
            "mobile": data.mobile,
            "email": data.email,
            "business_name": data.business_name,
            "region": data.region,
            "password_hash": hash_password(data.password),
        }).execute()
        print("Insert result:", result)
    except Exception as e:
        print("Insert error:", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "message": f"Account created for {data.name}", "user_id": result.data[0]["id"]}


@app.post("/login")
def login(data: LoginData):
    hashed = hash_password(data.password)

    result = supabase.table("artisans").select("*") \
        .or_(f"email.eq.{data.identifier},mobile.eq.{data.identifier}") \
        .execute()

    if not result.data:
        raise HTTPException(status_code=401, detail="No account found with this email/mobile.")

    user = result.data[0]

    if user["password_hash"] != hashed:
        raise HTTPException(status_code=401, detail="Incorrect password.")

    return {
        "success": True,
        "message": f"Welcome back, {user['name']}!",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "business_name": user["business_name"],
        },
    }


@app.post("/products")
def add_product(data: ProductData):
    result = supabase.table("products").insert({
        "artisan_id": data.artisan_id,
        "title": data.title,
        "description": data.description,
        "category": data.category,
        "price": data.price,
    }).execute()

    return {"success": True, "product": result.data[0]}


@app.get("/products/{artisan_id}")
def get_products(artisan_id: str):
    result = supabase.table("products").select("*").eq("artisan_id", artisan_id).order("created_at", desc=True).execute()
    return {"products": result.data}


@app.delete("/products/{product_id}")
def delete_product(product_id: str):
    supabase.table("products").delete().eq("id", product_id).execute()
    return {"success": True}


@app.post("/pricing")
def calculate_pricing(data: PricingData):
    total_cost = data.raw_material_cost + data.labour_cost + data.packaging_cost

    category_key = data.category.strip().lower()
    margin_rate = CATEGORY_MARGINS.get(category_key, DEFAULT_MARGIN)

    suggested_price = round(total_cost * (1 + margin_rate), 2)
    margin_amount = round(suggested_price - total_cost, 2)

    explanation = (
        f"Total cost = ₹{data.raw_material_cost} (materials) + ₹{data.labour_cost} (labour) + "
        f"₹{data.packaging_cost} (packaging) = ₹{total_cost}. "
        f"A {int(margin_rate * 100)}% margin was applied "
        f"({'category default for ' + category_key if category_key in CATEGORY_MARGINS else 'general default, since category was not recognized'}), "
        f"adding ₹{margin_amount}. Suggested selling price: ₹{suggested_price}. "
        f"This is a transparent rule-based calculation, not a machine learning prediction."
    )

    return {
        "total_cost": total_cost,
        "margin_rate": margin_rate,
        "margin_amount": margin_amount,
        "suggested_price": suggested_price,
        "explanation": explanation,
    }