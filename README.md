# Churn Prediction API

A production-ready FastAPI application for customer churn prediction using Machine Learning with Cloudflare R2 storage and MongoDB integration.

## 📁 Project Structure

```
churn-prediction-api/
├── main.py              # FastAPI application and endpoints
├── models.py            # ML models and prediction logic
├── database.py          # MongoDB configuration and helpers
├── schemas.py           # Pydantic models for request/response
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (not in git)
```

## 🚀 Features

- **Separation of concerns**: ML models, API endpoints, and database logic in separate modules
- **Cloud storage**: CSV files stored in Cloudflare R2 buckets for scalability and reliability
- **MongoDB integration**: Store prediction results and file metadata
- **Dual ML models**: 
  - Logistic Regression (fast baseline model)
  - Random Forest Classifier (ensemble model for better accuracy)
- **Risk analysis**: Identify at-risk users based on churn probability thresholds
- **Model evaluation**: Comprehensive metrics (accuracy, precision, recall, F1-score)
- **Async operations**: Fast, non-blocking API using async/await
- **Type safety**: Full Pydantic validation for requests/responses
- **Production ready**: CORS enabled, error handling, health checks

## 📋 Prerequisites

- Python 3.8+
- MongoDB (local or MongoDB Atlas)
- Cloudflare R2 account with bucket created

## 🛠️ Installation

### 1. Clone or create the project directory

```bash
mkdir churn-prediction-api
cd churn-prediction-api
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```bash
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=churn_prediction_db

# Storage Configuration
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=your_bucket_name
R2_ENDPOINT_URL=<endpoint_url>
```

**For MongoDB Atlas (cloud):**
```bash
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/
```

## ▶️ Running the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**API will be available at:** `http://localhost:8000`

**Interactive documentation:** `http://localhost:8000/docs`

## 📡 API Endpoints

### Health Check

#### `GET /`
Check API health status

**Response:**
```json
{
  "status": "running",
  "message": "Churn Prediction API is running",
  "version": "1.0.0",
  "storage": "Cloudflare R2",
  "database": "MongoDB"
}
```

---

### File Management

#### `POST /upload`
Upload CSV file to Cloudflare R2 bucket

**Request:**
- Content-Type: `multipart/form-data`
- Body: CSV file

**Response:**
```json
{
  "filename": "uploaded_files/uuid_filename.csv",
  "original_name": "customer_data.csv",
  "rows": 5000,
  "columns": 15,
  "file_size": 245760,
  "file_id": "507f1f77bcf86cd799439011"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@customer_data.csv"
```

---

#### `GET /files`
List all uploaded files with metadata

**Response:**
```json
{
  "files": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "filename": "uploaded_files/uuid_filename.csv",
      "original_name": "customer_data.csv",
      "file_size": 245760,
      "rows": 5000,
      "columns": 15,
      "status": "completed",
      "upload_date": "2025-09-30T10:30:00"
    }
  ]
}
```

---

### Model Training

#### `POST /train`
Train ML models and generate churn predictions

**Request Body:**
```json
{
  "filename": "uploaded_files/uuid_filename.csv"
}
```

**Response:**
```json
{
  "success": true,
  "filename": "uploaded_files/uuid_filename.csv",
  "result_filename": "results/uuid_results.csv",
  "num_predictions": 5000,
  "predictions": [
    {
      "CustomerID": "C001",
      "LogisticRegressionProb": 0.75,
      "RandomForestProb": 0.82,
      "ChurnPrediction": 1
    }
  ],
  "metrics": {
    "logistic_regression": {
      "accuracy": 0.85,
      "precision": 0.83,
      "recall": 0.78,
      "f1_score": 0.80
    },
    "random_forest": {
      "accuracy": 0.89,
      "precision": 0.87,
      "recall": 0.84,
      "f1_score": 0.85
    }
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"filename": "uploaded_files/uuid_filename.csv"}'
```

---

### Risk Analysis

#### `POST /at-risk-users`
Get users above specified churn risk threshold

**Request Body:**
```json
{
  "filename": "uploaded_files/uuid_filename.csv",
  "threshold": 0.7
}
```

**Response:**
```json
{
  "filename": "uploaded_files/uuid_filename.csv",
  "threshold": 0.7,
  "at_risk_count": 342,
  "at_risk_users": [
    {
      "CustomerID": "C001",
      "LogisticRegressionProb": 0.85,
      "RandomForestProb": 0.78,
      "ChurnPrediction": 1
    }
  ]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/at-risk-users" \
  -H "Content-Type: application/json" \
  -d '{"filename": "uploaded_files/uuid_filename.csv", "threshold": 0.7}'
```

---

### Model Evaluation

#### `POST /evaluate`
Evaluate model performance with detailed metrics

**Request Body:**
```json
{
  "filename": "uploaded_files/uuid_filename.csv"
}
```

**Response:**
```json
{
  "logistic_regression": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.78,
    "f1_score": 0.80,
    "confusion_matrix": [[850, 150], [110, 390]]
  },
  "random_forest": {
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.84,
    "f1_score": 0.85,
    "confusion_matrix": [[880, 120], [80, 420]]
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"filename": "uploaded_files/uuid_filename.csv"}'
```

---

### Results Retrieval

#### `GET /results/{filename}`
Get prediction results for a specific file from MongoDB

**Parameters:**
- `filename`: File identifier (e.g., "uploaded_files/uuid_filename.csv")

**Response:**
```json
{
  "_id": "507f1f77bcf86cd799439011",
  "filename": "uploaded_files/uuid_filename.csv",
  "original_filename": "customer_data.csv",
  "predictions": [...],
  "metrics": {...},
  "created_at": "2025-09-30T10:35:00"
}
```

**Example:**
```bash
curl "http://localhost:8000/results/uploaded_files/uuid_filename.csv"
```

---

#### `GET /results`
Get all prediction results from MongoDB

**Response:**
```json
{
  "count": 5,
  "results": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "filename": "uploaded_files/uuid_filename.csv",
      "predictions": [...],
      "metrics": {...}
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:8000/results"
```

---

## 🤖 Machine Learning Models

### 1. Logistic Regression
- Fast, interpretable baseline model
- Best for linear relationships
- Provides probability estimates
- Lower computational cost

### 2. Random Forest Classifier
- Ensemble learning method (100 decision trees)
- Handles non-linear patterns effectively
- Feature importance analysis
- Generally higher accuracy
- More robust to outliers

## 🐛 Troubleshooting

**MongoDB connection issues:**
```bash
# Check if MongoDB is running
mongosh

# Verify connection string
echo $MONGODB_URL
```

**Model training errors:**
- Ensure CSV has required columns
- Check for missing values in target column
- Minimum 100 rows recommended for training
- Verify 'Churn' column exists for evaluation

**File not found errors:**
- Check filename includes correct prefix
- Verify file exists in R2 bucket
- Use exact filename from upload response

## 📊 Example Workflow

```bash
# 1. Upload CSV file
curl -X POST "http://localhost:8000/upload" \
  -F "file=@customer_data.csv"

# Response: { "filename": "uploaded_files/uuid_123.csv", ... }

# 2. Train models
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"filename": "uploaded_files/uuid_123.csv"}'

# 3. Get at-risk users
curl -X POST "http://localhost:8000/at-risk-users" \
  -H "Content-Type: application/json" \
  -d '{"filename": "uploaded_files/uuid_123.csv", "threshold": 0.75}'

# 4. Retrieve results
curl "http://localhost:8000/results/uploaded_files/uuid_123.csv"
```
