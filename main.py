import os
import base64
import uuid
import json
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv
import hashlib
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageOps

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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
    image_url: str = ""


class PricingData(BaseModel):
    raw_material_cost: float
    labour_cost: float
    packaging_cost: float
    category: str = ""


class ImageUploadData(BaseModel):
    image_base64: str
    enhance: bool = True


class CatalogueInput(BaseModel):
    text: str
    language: str = "Hindi"


class VoiceCatalogueInput(BaseModel):
    audio_base64: str
    mime_type: str = "audio/m4a"
    language: str = "Hindi"


class ArtisanPhotoData(BaseModel):
    artisan_id: str
    image_base64: str


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

LANGUAGE_SCRIPT_HINTS = {
    "hindi": "Devanagari script",
    "marathi": "Devanagari script (Marathi wording and grammar, NOT Hindi wording)",
    "bengali": "Bengali script",
    "tamil": "Tamil script",
    "telugu": "Telugu script",
    "gujarati": "Gujarati script",
    "kannada": "Kannada script",
    "malayalam": "Malayalam script",
    "punjabi": "Gurmukhi script",
    "odia": "Odia script",
    "assamese": "Assamese script (Bengali-based script)",
    "urdu": "Urdu script (Nastaliq/Arabic-based script)",
}

CATALOGUE_PROMPT_TEMPLATE = """You are helping an Indian artisan create a bilingual product listing.
{source_instruction}

CRITICAL INSTRUCTION: The artisan's chosen language for the regional fields is "{language}".
You MUST write "title_regional" and "description_regional" in {language} language specifically, using {script_hint}.
Do NOT default to Hindi unless {language} literally is Hindi. If you are unsure how to write in {language}, still make your best genuine attempt in that language's script and grammar - never silently substitute a different language.

Respond with ONLY a valid JSON object, no markdown formatting, no code fences, in exactly this structure:

{{
  "title_english": "short catchy English product title",
  "title_regional": "short catchy title, written specifically in {language} ({script_hint})",
  "description_english": "2-3 sentence English product description, appealing to online buyers",
  "description_regional": "2-3 sentence description, written specifically in {language} ({script_hint})",
  "category": "one of: Textiles, Pottery, Jewelry, Woodwork, Leather, Paintings, Handicrafts, Other",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "seo_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "regional_language": "{language}"
}}"""


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def parse_gemini_json(raw_text: str):
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    return json.loads(raw_text)


def build_catalogue_prompt(source_instruction: str, language: str) -> str:
    script_hint = LANGUAGE_SCRIPT_HINTS.get(language.strip().lower(), f"{language}'s native script")
    return CATALOGUE_PROMPT_TEMPLATE.format(
        source_instruction=source_instruction,
        language=language,
        script_hint=script_hint,
    )


def enhance_image(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    img = ImageOps.autocontrast(img, cutoff=1)

    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(1.08)

    color = ImageEnhance.Color(img)
    img = color.enhance(1.12)

    sharpness = ImageEnhance.Sharpness(img)
    img = sharpness.enhance(1.2)

    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))

    img.thumbnail((1080, 1080))

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=88)
    return output.getvalue()


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


@app.get("/artisan/{artisan_id}")
def get_artisan_profile(artisan_id: str):
    result = supabase.table("artisans").select("id, name, mobile, email, business_name, region, photo_url, created_at").eq("id", artisan_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Artisan not found.")
    return {"artisan": result.data[0]}


@app.post("/artisan-photo")
def update_artisan_photo(data: ArtisanPhotoData):
    try:
        image_bytes = base64.b64decode(data.image_base64)
        image_bytes = enhance_image(image_bytes)

        filename = f"artisan-{data.artisan_id}.jpg"

        supabase.storage.from_("product-images").upload(
            filename, image_bytes, {"content-type": "image/jpeg", "upsert": "true"}
        )

        public_url = supabase.storage.from_("product-images").get_public_url(filename)

        supabase.table("artisans").update({"photo_url": public_url}).eq("id", data.artisan_id).execute()

        return {"success": True, "photo_url": public_url}
    except Exception as e:
        print("Artisan photo error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-image")
def upload_image(data: ImageUploadData):
    try:
        image_bytes = base64.b64decode(data.image_base64)

        if data.enhance:
            image_bytes = enhance_image(image_bytes)

        filename = f"{uuid.uuid4()}.jpg"

        supabase.storage.from_("product-images").upload(
            filename, image_bytes, {"content-type": "image/jpeg"}
        )

        public_url = supabase.storage.from_("product-images").get_public_url(filename)

        return {"success": True, "image_url": public_url}
    except Exception as e:
        print("Upload error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/products")
def add_product(data: ProductData):
    result = supabase.table("products").insert({
        "artisan_id": data.artisan_id,
        "title": data.title,
        "description": data.description,
        "category": data.category,
        "price": data.price,
        "image_url": data.image_url,
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


@app.post("/generate-catalogue")
def generate_catalogue(data: CatalogueInput):
    source_instruction = f'The artisan described their product as:\n\n"{data.text}"'
    prompt = build_catalogue_prompt(source_instruction, data.language)

    try:
        model = genai.GenerativeModel("gemini-3.8-flash")
        response = model.generate_content(prompt)
        parsed = parse_gemini_json(response.text)
        return {"success": True, "catalogue": parsed}
    except Exception as e:
        print("Gemini error:", e)
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@app.post("/generate-catalogue-voice")
def generate_catalogue_voice(data: VoiceCatalogueInput):
    source_instruction = "The artisan described their product by speaking, in the audio clip provided."
    prompt = build_catalogue_prompt(source_instruction, data.language)

    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        model = genai.GenerativeModel("gemini-3.8-flash")
        response = model.generate_content([
            {"mime_type": data.mime_type, "data": audio_bytes},
            prompt,
        ])
        parsed = parse_gemini_json(response.text)
        return {"success": True, "catalogue": parsed}
    except Exception as e:
        print("Gemini voice error:", e)
        raise HTTPException(status_code=500, detail=f"AI voice generation failed: {str(e)}")


@app.get("/marketplace")
def get_marketplace_products():
    products_result = supabase.table("products").select("*").order("created_at", desc=True).execute()
    products = products_result.data

    artisan_ids = list(set(p["artisan_id"] for p in products if p.get("artisan_id")))
    artisans_map = {}

    if artisan_ids:
        artisans_result = supabase.table("artisans").select("id, name, business_name, region").in_("id", artisan_ids).execute()
        for a in artisans_result.data:
            artisans_map[a["id"]] = a

    for p in products:
        p["artisan_info"] = artisans_map.get(p.get("artisan_id"))

    return {"products": products}