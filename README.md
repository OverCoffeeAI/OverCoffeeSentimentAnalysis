# Sentiment Analysis API

A FastAPI-based sentiment analysis service that classifies text as POSITIVE, NEGATIVE, or NEUTRAL with confidence scores.

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/OverCoffeeAI/YourRepoName.git
cd YourRepoName
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv sentiment_env

# Activate virtual environment
# On Mac/Linux:
source sentiment_env/bin/activate

# On Windows:
sentiment_env\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install pandas numpy tensorflow scikit-learn joblib fastapi uvicorn
```

### 4. Run the API

```bash
python main.py
```

The API will be available at: **http://localhost:8000**

## 📋 API Usage

### Interactive Documentation

Visit **http://localhost:8000/docs** for Swagger UI documentation.

### Endpoints

#### POST `/analyze`

Analyze sentiment of text.

**Request:**

```json
{
  "message": "I love this new product!"
}
```

**Response:**

```json
{
  "sentiment": "POSITIVE",
  "confidence": 0.847
}
```
