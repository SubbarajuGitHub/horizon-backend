# Churn Prediction API

A production-ready FastAPI application for customer churn prediction using Machine Learning.

## 📁 Project Structure

```
churn-prediction-api/
├── main.py              # FastAPI application and endpoints
├── models.py            # ML models and prediction logic
├── database.py          # MongoDB configuration and helpers
├── schemas.py           # Pydantic models for request/response
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── uploads/             # Temporary file storage (created automatically)
└── results/             # Model files storage (created automatically)
```

## 🚀 Features

- **Separate concerns**: ML models, API endpoints, and database logic are in separate files
- **MongoDB integration**: Store file metadata and prediction results
- **Efficient storage**: Store only file metadata in DB, actual CSV files on disk
- **Async operations**: Fast, non-blocking API using async/await
- **Type safety**: Full Pydantic validation for requests/responses
- **Production ready**: CORS enabled, error handling, health checks

## 📋 Prerequisites

- Python 3.8+
- MongoDB (local or Atlas)

## 🛠️ Installation

1. **Clone or create the project directory**

```bash
mkdir churn-prediction-api
cd churn-prediction-api
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file or use default values:

```bash
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=churn_prediction_db
```

For MongoDB Atlas (cloud):
```bash
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/
```

5. **Start MongoDB** (if running locally)

```bash
mongod
```

## ▶️ Running the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

## 📡 API Endpoints

### File Management

- **POST /upload** - Upload CSV file
- **GET /files** - List all uploaded files

### Predictions

- **POST /predict** - Train models and get predictions
- **POST /at-risk-users** - Get users above risk threshold
- **POST /evaluate** - Evaluate model performance

### Results

- **GET /results/{filename}** - Get prediction results from MongoDB

## 💾 Storage Strategy

### Why This Approach?

1. **CSV Files on Disk**: 
   - Fast read/write operations
   - No size limitations
   - Easy to manage and delete

2. **Metadata in MongoDB**:
   - File information (name, size, upload date)
   - Prediction summaries (metrics, high-risk count)
   - Processing status tracking

3. **Best of Both Worlds**:
   - MongoDB for searchable metadata
   - Disk for large data files
   - No performance issues with large CSVs

## 🚢 Deployment Options

### Option 1: Cloud VM (Recommended)

Deploy on AWS EC2, Google Cloud, or DigitalOcean:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with gunicorn for production
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Files will be stored on the VM's disk, persistent across requests.

### Option 2: Docker

```dockerfile
FROM python:3.9-slim