"""
Professional Admission Inquiry Chatbot - FastAPI Backend
Author: College AI Team
Features: Intent classification, conversation tracking, analytics
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import os
from datetime import datetime
import logging
from collections import defaultdict
import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Admission Inquiry Chatbot API",
    description="Professional chatbot API for college admission inquiries",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
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

# Analytics storage (in production, use a database)
analytics = {
    'total_queries': 0,
    'intent_counts': defaultdict(int),
    'conversations': []
}


# Pydantic models
class Query(BaseModel):
    text: str
    conversation_id: Optional[str] = None


class PredictionResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    timestamp: str
    conversation_id: Optional[str] = None


class AnalyticsResponse(BaseModel):
    total_queries: int
    intent_distribution: dict
    average_confidence: float


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[dict]
    started_at: str
    last_updated: str


def load_model(model_path=None):
    """Load trained model and configurations"""
    global model, tokenizer, label_mapping, responses, device
    
    model_path = model_path or config.MODEL_DIR
    
    logger.info(f"Loading model from {model_path}...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Load label mapping
        with open(f'{model_path}/label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
        # Load responses
        responses_path = f'{model_path}/responses.json'
        if os.path.exists(responses_path):
            with open(responses_path, 'r') as f:
                responses = json.load(f)
        else:
            responses = config.RESPONSES
            with open(responses_path, 'w') as f:
                json.dump(responses, f, indent=2)
        
        logger.info("✓ Model loaded successfully!")
        logger.info(f"✓ Loaded {len(label_mapping)} intents")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def predict_intent(text: str):
    """Predict intent from user input with confidence score"""
    try:
        # Tokenize
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
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        intent = label_mapping[str(predicted_class)]
        
        logger.info(f"Query: '{text[:50]}...' | Intent: {intent} | Confidence: {confidence:.4f}")
        
        return intent, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


def get_response(intent: str) -> str:
    """Get response text for predicted intent"""
    return responses.get(intent, responses.get("greeting", "I'm here to help with your admission queries!"))


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting Admission Inquiry Chatbot API...")
    load_model()
    logger.info("API ready to serve requests!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    # Save analytics (in production, save to database)
    with open('analytics_data.json', 'w') as f:
        json.dump(analytics, f, indent=2)
    logger.info("Analytics saved. Goodbye!")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "College Admission Inquiry Chatbot API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(query: Query):
    """
    Predict intent and return response for user query
    
    - **text**: User's question or message
    - **conversation_id**: Optional conversation identifier
    """
    try:
        if not query.text or not query.text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        # Predict intent
        intent, confidence = predict_intent(query.text.strip())
        
        # Get response
        response_text = get_response(intent)
        
        # Update analytics
        analytics['total_queries'] += 1
        analytics['intent_counts'][intent] += 1
        
        # Create response
        result = PredictionResponse(
            intent=intent,
            confidence=confidence,
            response=response_text,
            timestamp=datetime.now().isoformat(),
            conversation_id=query.conversation_id
        )
        
        # Log conversation (in production, save to database)
        analytics['conversations'].append({
            'query': query.text,
            'intent': intent,
            'confidence': confidence,
            'timestamp': result.timestamp,
            'conversation_id': query.conversation_id
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/intents")
async def get_intents():
    """Get all available intents"""
    if not label_mapping:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "intents": list(label_mapping.values()),
        "count": len(label_mapping)
    }


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get analytics data"""
    total = analytics['total_queries']
    
    if total == 0:
        avg_confidence = 0.0
    else:
        total_confidence = sum(
            conv['confidence'] 
            for conv in analytics['conversations']
        )
        avg_confidence = total_confidence / total
    
    return AnalyticsResponse(
        total_queries=total,
        intent_distribution=dict(analytics['intent_counts']),
        average_confidence=avg_confidence
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "total_queries_served": analytics['total_queries']
    }


@app.post("/feedback")
async def submit_feedback(feedback: dict):
    """Submit user feedback"""
    try:
        # In production, save to database
        logger.info(f"Feedback received: {feedback}")
        return {"message": "Feedback received. Thank you!"}
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")


@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    model_path = config.MODEL_DIR
    metadata_path = f'{model_path}/training_metadata.json'
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    else:
        return {
            "message": "Model metadata not available",
            "model_loaded": model is not None
        }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower()
    )