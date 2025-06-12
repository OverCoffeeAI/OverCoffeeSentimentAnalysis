from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple API to analyze sentiment of text messages",
    version="1.0.0"
)

# Load models at startup (load once, use many times)
try:
    model = tf.keras.models.load_model('model.h5')
    tokenizer = joblib.load('tokenizer.joblib')
    encoder = joblib.load('encoder.joblib')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    tokenizer = None
    encoder = None

# Request model
class SentimentRequest(BaseModel):
    message: str

# Response model
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

# Function to predict sentiment
def predict_sentiment(text: str) -> tuple[str, float]:
    if not all([model, tokenizer, encoder]):
        raise HTTPException(status_code=500, detail="Models not loaded properly")
    
    # Clean text
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove common greetings that bias the model
    text = re.sub(r'\b(hello|hi|hey|good morning|good afternoon)\b', '', text, flags=re.IGNORECASE)
    
    text = text.lower().strip()
    
    # Process
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=100, padding='post')
    
    # Predict
    pred = model.predict(pad, verbose=0)[0]
    confidence = float(pred.max())  # Convert to float for JSON serialization
    
    # If confidence is below 0.65, classify as neutral
    if confidence < 0.65:
        sentiment = "NEUTRAL"
    else:
        raw_sentiment = encoder.inverse_transform([pred.argmax()])[0]
        sentiment = raw_sentiment.upper()  # Convert to uppercase
    
    return sentiment, confidence

# API Endpoints

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running!"}

# Endpoint to analyze sentiment
@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        sentiment, confidence = predict_sentiment(request.message)
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Health check endpoint to verify models are loaded
@app.get("/health")
async def health_check():
    models_loaded = all([model, tokenizer, encoder])
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    }

# Run the app 
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes 
    )