import os
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import fitz
from PIL import Image
import io
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib
from app.routes import property, user, ml
from app.services.auth import router as auth_router
from app.models.database import Base, async_engine
from app.config import IMAGE_SOURCE_DIR, logger
from app.services.google_maps import estimate_sqft_from_google_earth, get_condition_from_street_view
from app.services.ml_predictor import predict_price
from app.services.image_analyzer import analyze_image
from design_style_model import RealEstateImageModel  # Import the updated image model

app = FastAPI(title="Raise Backend", description="Real estate valuation and analysis for Raise Inc., Brokerage")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.access(IMAGE_SOURCE_DIR, os.W_OK):
    logger.warning(f"IMAGE_SOURCE_DIR {IMAGE_SOURCE_DIR} is not writable. Using /tmp/raise_images instead.")
    IMAGE_SOURCE_DIR = "/tmp/raise_images"
os.makedirs(IMAGE_SOURCE_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=IMAGE_SOURCE_DIR), name="static")

# Load ML models
print("Loading ML models...")
DESIGN_TREND_MODEL_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "design_trend_classifier.pkl")
try:
    design_trend_classifier = joblib.load(DESIGN_TREND_MODEL_PATH)
    print(f"Loaded design trend classifier from {DESIGN_TREND_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load design trend classifier: {e}")
    design_trend_classifier = None

mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Initialize RealEstateImageModel
image_model = RealEstateImageModel()
image_model.setup()

# Function to extract features from an image (used for design trend classifier)
def extract_image_features(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img)
        if img_array.shape[-1] != 3:  # Ensure RGB
            img_array = np.stack([img_array] * 3, axis=-1) if len(img_array.shape) == 2 else img_array[:, :, :3]
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = mobilenet_model.predict(img_array)
        return features.flatten()
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return np.zeros((1280,))

# Enhanced PDF processing with ML predictions
async def extract_and_analyze_pdf(pdf_file: UploadFile) -> dict:
    images = []
    text_content = ""
    try:
        pdf_bytes = await pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Extract text
        for page in doc:
            text_content += page.get_text()
        
        # Extract and analyze images
        image_features = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))
                save_path = os.path.join(IMAGE_SOURCE_DIR, f"page_{page_num+1}_img_{img_index}.{image_ext}")
                image.save(save_path)
                
                # Basic image analysis (from app.services.image_analyzer)
                basic_analysis = analyze_image(save_path)
                
                # Extract design trend using existing classifier
                features = extract_image_features(save_path)
                design_trend = design_trend_classifier.predict([features])[0] if design_trend_classifier else "Unknown"
                
                # Analyze image using RealEstateImageModel
                task = {'data': {'image': save_path}}
                prediction = image_model.predict([task])[0]['result']
                detailed_analysis = {
                    'room_type': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'room_type'),
                    'quality_of_finishes': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'quality_of_finishes'),
                    'recency_of_renovation': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'recency_of_renovation'),
                    'lighting': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'lighting'),
                    'layout': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'layout'),
                    'condition': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'condition'),
                    'desirable_feature': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'desirable_feature'),
                    'undesirable_feature': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'undesirable_feature'),
                    'room_size': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'room_size'),
                    'room_functionality': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'room_functionality'),
                    'appliance_quality': next(item['value']['choices'][0] for item in prediction if item['from_name'] == 'appliance_quality'),
                }
                
                image_info = {
                    "page": page_num + 1,
                    "index": img_index,
                    "file_path": save_path,
                    "url": f"/static/page_{page_num+1}_img_{img_index}.{image_ext}",
                    "analysis": {**basic_analysis, "design_trend": design_trend, **detailed_analysis}
                }
                images.append(image_info)
                image_features.append(detailed_analysis)
                logger.info(f"Extracted and analyzed image: {save_path}")

        doc.close()
        
        # ML predictions from text and image features
        import re
        sqft_match = re.search(r"SQUARE FEET\s*(\d+)-(\d+)|(\d+)\s*sqft", text_content, re.IGNORECASE)
        sqft = (int(sqft_match.group(1)) + int(sqft_match.group(2))) / 2 if sqft_match and sqft_match.group(1) else (int(sqft_match.group(3)) if sqft_match and sqft_match.group(3) else 1500)
        beds = int(re.search(r"BEDS\s*(\d+)", text_content, re.IGNORECASE).group(1)) if re.search(r"BEDS\s*(\d+)", text_content, re.IGNORECASE) else 3
        baths = int(re.search(r"BATHS?\s*(\d+)", text_content, re.IGNORECASE).group(1)) if re.search(r"BATHS?\s*(\d+)", text_content, re.IGNORECASE) else 1
        text_lower = text_content.lower()
        finish_quality = "high" if "modern" in text_lower or "renovated" in text_lower else ("low" if "no updates" in text_lower or "as-is" in text_lower else "medium")
        reno_potential = "Yes" if "unfinished" in text_lower or "needs updating" in text_lower else "No"
        
        # Aggregate image features (using the first image for simplicity; could average across images)
        avg_image_features = image_features[0] if image_features else {
            'quality_of_finishes': 'medium',
            'recency_of_renovation': 'modern',
            'condition': 'move-in ready',
            'desirable_feature': 'present',
            'undesirable_feature': 'absent',
            'appliance_quality': 'standard'
        }

        features = {
            'beds': beds,
            'baths': baths,
            'sqft': sqft,
            'quality': 0.9 if avg_image_features['quality_of_finishes'] in ['high', 'mid-high'] else (0.5 if avg_image_features['quality_of_finishes'] in ['low', 'mid-low'] else 0.7),
            'reno_potential': 1 if reno_potential == "Yes" else 0,
            'hardwood': 1 if "hardwood" in text_lower else 0,
            'quartz': 1 if "quartz" in text_lower else 0,
            'ss_appliances': 1 if "stainless" in text_lower or avg_image_features['appliance_quality'] == 'high-end' else 0,
            'condition': 0.9 if avg_image_features['condition'] == 'move-in ready' else (0.5 if avg_image_features['condition'] == 'needs major renovation' else 0.7),
            'desirable_feature': 1 if avg_image_features['desirable_feature'] == 'present' else 0,
            'undesirable_feature': 1 if avg_image_features['undesirable_feature'] == 'present' else 0,
        }
        predicted_price = predict_price(features)
        
        return {
            "images": images,
            "text_analysis": {
                "sqft": sqft,
                "beds": beds,
                "baths": baths,
                "finish_quality": avg_image_features['quality_of_finishes'],
                "reno_potential": reno_potential,
                "condition": avg_image_features['condition'],
                "predicted_price": round(predicted_price)
            }
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

@app.post("/api/extract-images/")
async def process_pdf_upload(file: UploadFile = File(...)):
    try:
        result = await extract_and_analyze_pdf(file)
        return {
            "status": "success",
            "images": [
                {"page": img["page"], "index": img["index"], "url": img["url"], "analysis": img["analysis"]}
                for img in result["images"]
            ],
            "text_analysis": result["text_analysis"]
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

# Existing Google Maps endpoint (unchanged for now)
@app.post("/api/google-maps-estimate/")
async def google_maps_estimate(request: dict):
    try:
        address = request.get("address", "Unknown Address")
        sqft = estimate_sqft_from_google_earth(address)
        condition = get_condition_from_street_view(address)

        features = {
            'beds': request.get('beds', 2),
            'baths': request.get('baths', 1),
            'rooms': request.get('rooms', 5),
            'dom': request.get('dom', 30),
            'age': request.get('age', 27),
            'quality': condition,
            'desirability': condition - 0.1 if "alley" in address.lower() else condition,
            'exposure_north': 1 if request.get('exposure', 'north') == 'north' else 0,
            'exposure_south': 1 if request.get('exposure', 'north') == 'south' else 0,
            'hardwood': 1 if request.get('hardwood', True) else 0,
            'quartz': 1 if request.get('quartz', True) else 0,
            'ss_appliances': 1 if request.get('ss_appliances', True) else 0,
            'inventory': request.get('inventory', 1.5),
            'sales_rate': request.get('sales_rate', 0.75),
            'trend': request.get('trend', -0.05),
            'sqft': sqft
        }
        
        predicted_price = predict_price(features)
        
        return {
            "status": "success",
            "address": address,
            "estimated_sqft": round(sqft),
            "condition_score": condition,
            "predicted_price": round(predicted_price)
        }
    except Exception as e:
        logger.error(f"Google Maps estimate error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.on_event("startup")
async def startup():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

app.include_router(property.router, prefix="/api")
app.include_router(user.router, prefix="/api")
app.include_router(ml.router, prefix="/api")
app.include_router(auth_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)