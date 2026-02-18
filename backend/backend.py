"""
Backend with Smart Response Filtering
Extracts only relevant sections when user asks about specific facilities
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import os
from datetime import datetime
import logging
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Admission Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
tokenizer = None
label_mapping = None
responses = None
device = None

class Query(BaseModel):
    text: str
    conversation_id: Optional[str] = None

class PredictionResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    timestamp: str


def extract_relevant_section(full_response, user_query):
    """
    Extract only relevant section from response based on user query
    """
    query_lower = user_query.lower()
    
    # Define keywords for each section
    section_keywords = {
        'library': ['library', 'book', 'reading', 'study room', 'e-book'],
        'hostel_info': ['hostel', 'accommodation', 'room', 'warden', 'residential'],
        'mess': ['mess', 'food', 'cafeteria', 'canteen', 'dining', 'meal'],
        'sports': ['sports', 'gym', 'fitness', 'cricket', 'football', 'basketball', 'swimming', 'playground'],
        'transport': ['transport', 'bus', 'parking', 'vehicle', 'shuttle'],
        'medical': ['medical', 'health', 'doctor', 'hospital', 'clinic', 'ambulance'],
        'lab': ['lab', 'laboratory', 'equipment', 'practical', 'experiment'],
        'classroom': ['classroom', 'smart class', 'lecture hall', 'seminar'],
        'wifi': ['wifi', 'internet', 'connectivity', 'network', 'online'],
        'security': ['security', 'cctv', 'safety', 'guard'],
        'banking': ['bank', 'atm', 'financial'],
        'recreation': ['recreation', 'entertainment', 'games', 'tv room']
    }
    
    # Check which specific facility is asked
    matched_sections = []
    for section, keywords in section_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            matched_sections.append(section)
    
    # If specific facility found, extract that section
    if matched_sections:
        extracted = extract_sections_from_text(full_response, matched_sections)
        if extracted:
            return extracted
    
    # Check for general queries
    general_keywords = ['what facilities', 'all facilities', 'available facilities', 'facilities available']
    if any(keyword in query_lower for keyword in general_keywords):
        return create_facilities_summary(full_response)
    
    # Default: return full response
    return full_response


def extract_sections_from_text(response, section_names):
    """
    Extract specific sections from the formatted response
    """
    lines = response.split('\n')
    extracted_lines = []
    capturing = False
    captured_anything = False
    
    # Section headers to look for
    section_headers = {
        'library': ['Library', '📚', 'LIBRARY'],
        'hostel_info': ['Hostel', '🏠', 'RESIDENTIAL'],
        'mess': ['Mess', 'Cafeteria', '🍽️', 'MESS', 'Dining'],
        'sports': ['Sports', '🏃', 'SPORTS', 'Fitness', 'Gymnasium'],
        'transport': ['Transport', '🚌', 'TRANSPORT', 'Parking'],
        'medical': ['Medical', '🏥', 'Health'],
        'lab': ['Laboratory', 'Lab', '🔬', 'LABORATORY'],
        'classroom': ['Classroom', 'Smart Classroom'],
        'wifi': ['Internet', 'WiFi', '📡', 'IT &'],
        'security': ['Security', '🔒', 'SECURITY'],
        'banking': ['Banking', '🏪', 'ATM'],
        'recreation': ['Recreation', 'Entertainment']
    }
    
    for section in section_names:
        headers = section_headers.get(section, [])
        in_section = False
        section_content = []
        
        for i, line in enumerate(lines):
            # Check if we hit the section header
            if any(header in line for header in headers):
                in_section = True
                section_content.append(line)
                captured_anything = True
                continue
            
            # If in section, keep capturing until we hit another major section
            if in_section:
                # Stop at next major section (starts with ** or emoji or ###)
                if line.strip() and (line.strip().startswith('**🏛️') or 
                                     line.strip().startswith('**📚') or 
                                     line.strip().startswith('**🏠') or
                                     line.strip().startswith('**🏃') or
                                     line.strip().startswith('##')):
                    # Check if it's not part of current section
                    if not any(header in line for header in headers):
                        in_section = False
                        break
                
                section_content.append(line)
                
                # Stop after reasonable amount (to avoid capturing too much)
                if len(section_content) > 80:
                    break
        
        if section_content:
            extracted_lines.extend(section_content)
            extracted_lines.append('\n')  # Add separator
    
    if captured_anything:
        result = '\n'.join(extracted_lines)
        result += "\n\n💡 **Want to know about other facilities?** Just ask!"
        return result
    
    return None


def create_facilities_summary(full_response):
    """
    Create a brief summary of all facilities when user asks generally
    """
    summary = """
🏛️ **Campus Facilities Overview**

We offer comprehensive facilities including:

📚 **Academic:** Library (50,000+ books), Modern Labs, Smart Classrooms, Research Centers

🏠 **Residential:** Separate Hostels for Boys & Girls, AC/Non-AC rooms, 24/7 WiFi

🍽️ **Food:** Hygienic Mess, Cafeteria, Food Court, Veg/Non-veg options

🏃 **Sports:** Cricket Ground, Football Field, Basketball Courts, Gymnasium, Swimming Pool

🏥 **Health:** Medical Center (24/7), Ambulance, Health Insurance

🚌 **Transport:** College Buses, Parking facilities

📡 **IT:** High-speed WiFi, Computer Labs, 1 Gbps internet

🔒 **Security:** 24/7 Guards, 300+ CCTV Cameras, Biometric Entry

🏦 **Convenience:** Banking, ATM, Post Office, Bookstore

🌿 **Green Campus:** 50+ acres, Eco-friendly, Solar Power

**Want detailed information about any specific facility?**
Ask me about: Library, Hostel, Sports, Cafeteria, Transport, Labs, or any other facility!
"""
    return summary


def load_model(model_path=None):
    """Load trained model and configurations"""
    global model, tokenizer, label_mapping, responses, device
    
    model_path = model_path or config.MODEL_DIR
    logger.info(f"Loading model from {model_path}...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        with open(f'{model_path}/label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
       
            responses = config.RESPONSES
        
        logger.info("✓ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def predict_intent(text: str):
    """Predict intent from user input"""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config.MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    intent = label_mapping[str(predicted_class)]
    return intent, confidence


def get_response(intent: str) -> str:
    """Get response for intent"""
    return responses.get(intent, responses.get("greeting", "I'm here to help!"))


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting Smart Admission Chatbot API...")
    load_model()
    logger.info("API ready!")


@app.get("/")
async def root():
    return {
        "message": "Smart Admission Chatbot API",
        "version": "2.1.0",
        "features": ["Smart response filtering", "Context-aware answers"]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(query: Query):
    """Predict intent and return smart response"""
    try:
        if not query.text or not query.text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        # Predict intent
        intent, confidence = predict_intent(query.text.strip())
        
        # Get full response
        response_text = get_response(intent)
        
        # ✨ SMART FILTERING: Extract relevant section for facilities
        if intent == "facilities":
            response_text = extract_relevant_section(response_text, query.text)
        
        # Also apply smart filtering for other long responses
        if intent in ["courses", "placement", "hostel", "documents", "exam"]:
            # These might also benefit from smart filtering in future
            pass
        
        logger.info(f"Query: '{query.text[:50]}...' → Intent: {intent} ({confidence:.2f})")
        
        return PredictionResponse(
            intent=intent,
            confidence=confidence,
            response=response_text,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)