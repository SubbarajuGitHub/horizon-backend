from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os
import shutil
from typing import Optional
import uuid
import boto3
from io import BytesIO
from botocore.exceptions import ClientError

from models import ChurnPredictor
from database import (
    connect_to_mongo, 
    close_mongo_connection, 
    FileMetadata, 
    PredictionResult
)
from schemas import (
    FileUploadResponse,
    FileRequest,
    AtRiskRequest,
    ChurnPredictionResponse,
    FileListResponse,
    AtRiskUsersResponse,
    ModelPerformanceResponse
)

R2_ACCOUNT_ID = "7bd6729b4621d457621491ed5ac6af8d"
R2_ACCESS_KEY_ID = "14efa6db9398c3cd0398e16f110f9c38"
R2_SECRET_ACCESS_KEY = "4f4c21f5b3267d7a3626bdde4c0f40f2638e8efe14e48465bfdfc978a635b237"
R2_BUCKET_NAME = "horizon"
R2_ENDPOINT_URL = "https://7bd6729b4621d457621491ed5ac6af8d.r2.cloudflarestorage.com"

s3_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name='auto'
)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="ML API for customer churn prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory (store files temporarily)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize ML model
predictor = ChurnPredictor()


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    await connect_to_mongo()
    print("Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    await close_mongo_connection()
    print("Application shutdown complete")


@app.get("/", tags=["Health Check"])
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Churn Prediction API is running",
        "version": "1.0.0"
    }

@app.post("/upload", response_model=FileUploadResponse, tags=["File Management"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV file for churn prediction
    - File is stored in R2 bucket with 'uploaded_files/' prefix
    - Metadata is stored in MongoDB
    """
    try:
        # Validate file type
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV files are allowed"
            )

        # Generate unique filename with prefix
        unique_filename = f"uploaded_files/{uuid.uuid4()}_{file.filename}"

        # Read file into memory
        file_bytes = await file.read()
        file_size = len(file_bytes)
        
        # Create buffer for R2 upload
        upload_buffer = BytesIO(file_bytes)

        # Upload buffer to R2 with prefix
        s3_client.upload_fileobj(upload_buffer, R2_BUCKET_NAME, unique_filename)

        # Create a NEW buffer for pandas
        csv_buffer = BytesIO(file_bytes)
        df = pd.read_csv(csv_buffer)
        rows, columns = df.shape

        # Store metadata in MongoDB
        file_id = await FileMetadata.create(
            filename=unique_filename,
            original_name=file.filename,
            file_size=file_size,
            rows=rows,
            columns=columns
        )

        return FileUploadResponse(
            filename=unique_filename,
            original_name=file.filename,
            rows=rows,
            columns=columns,
            file_size=file_size,
            file_id=file_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )
    

@app.get("/files", response_model=FileListResponse, tags=["File Management"])
async def list_files():
    """List all uploaded files from MongoDB"""
    try:
        files = await FileMetadata.list_all()
        # Convert ObjectId to string
        for file in files:
            file['_id'] = str(file['_id'])
        return FileListResponse(files=files)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files: {str(e)}"
        )


@app.post("/predict", tags=["Prediction"])
async def predict_churn(request: FileRequest):
    """
    Train models and predict churn probabilities
    - Loads file from R2 bucket
    - Trains models
    - Returns predictions
    - Stores results in MongoDB and R2 (results/ prefix)
    """
    try:
        # Use filename as-is if it already has the prefix, otherwise add it
        if request.filename.startswith("uploaded_files/"):
            r2_file_key = request.filename
        else:
            r2_file_key = f"uploaded_files/{request.filename}"
        
        # Check if file exists in R2
        try:
            s3_client.head_object(Bucket=R2_BUCKET_NAME, Key=r2_file_key)
        except ClientError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {r2_file_key} not found in R2 bucket"
            )
        
        # Update file status (use the full key for consistency)
        await FileMetadata.update_status(r2_file_key, "processing")
        
        # Download file from R2 into memory
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=r2_file_key)
        file_bytes = response['Body'].read()
        
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_file:
            tmp_file.write(file_bytes)
            temp_file_path = tmp_file.name
        
        try:
            # Process file with ML model
            result = predictor.process_file(temp_file_path)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        if not result["success"]:
            await FileMetadata.update_status(r2_file_key, "failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        # Create results DataFrame
        results_df = pd.DataFrame(result["predictions"])
        
        # Generate unique result filename
        original_filename = os.path.basename(r2_file_key)
        result_filename = f"results/{uuid.uuid4()}_{original_filename}"
        
        # Convert DataFrame to CSV and upload to R2
        csv_buffer = BytesIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Upload results to R2
        s3_client.upload_fileobj(csv_buffer, R2_BUCKET_NAME, result_filename)
        
        await PredictionResult.create(
            filename=r2_file_key,
            original_filename=request.filename,
            predictions=result["predictions"],
            metrics=result["metrics"]
        )
        
        # Update file status
        await FileMetadata.update_status(r2_file_key, "completed")
        
        return ChurnPredictionResponse(
            success=True,
            filename=r2_file_key,
            result_filename=result_filename,
            num_predictions=len(result["predictions"]),
            predictions=result["predictions"],
            metrics=result["metrics"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await FileMetadata.update_status(r2_file_key if 'r2_file_key' in locals() else request.filename, "failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
         

@app.post("/at-risk-users", response_model=AtRiskUsersResponse, tags=["Prediction"])
async def get_at_risk_users(request: AtRiskRequest):
    """
    Get list of at-risk users based on threshold from R2 bucket
    """
    try:
        if request.filename.startswith("uploaded_files/"):
            r2_file_key = request.filename
        else:
            r2_file_key = f"uploaded_files/{request.filename}"


        # Check if file exists in R2
        try:
            s3_client.head_object(Bucket=R2_BUCKET_NAME, Key=r2_file_key)
        except ClientError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {r2_file_key} not found in R2 bucket"
            )

        # Download file from R2 into memory
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=r2_file_key)
        file_bytes = response['Body'].read()

        # Create temporary file to pass to predictor
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp_file:
            tmp_file.write(file_bytes)
            temp_file_path = tmp_file.name

        try:
            result = predictor.process_file(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        # Filter at-risk users
        at_risk_users = [
            pred for pred in result["predictions"]
            if pred['LogisticRegressionProb'] > request.threshold or 
               pred['RandomForestProb'] > request.threshold
        ]

        return AtRiskUsersResponse(
            filename=request.filename,
            threshold=request.threshold,
            at_risk_count=len(at_risk_users),
            at_risk_users=at_risk_users
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get at-risk users: {str(e)}"
        )


@app.post("/evaluate", response_model=ModelPerformanceResponse, tags=["Evaluation"])
async def evaluate_models(request: FileRequest):
    """
    Evaluate model performance on uploaded file from R2 bucket
    """
    try:
        if request.filename.startswith("uploaded_files/"):
            r2_file_key = request.filename
        else:
            r2_file_key = f"uploaded_files/{request.filename}"

        # Check if file exists in R2
        try:
            s3_client.head_object(Bucket=R2_BUCKET_NAME, Key=r2_file_key)
        except ClientError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {r2_file_key} not found in R2 bucket"
            )

        # Download file into memory
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=r2_file_key)
        file_bytes = response['Body'].read()

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp_file:
            tmp_file.write(file_bytes)
            temp_file_path = tmp_file.name

        try:
            df = pd.read_csv(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        if 'Churn' not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV must contain 'Churn' column for evaluation"
            )

        # Process file
        result = predictor.process_file(temp_file_path)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        if result["metrics"] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot evaluate without target variable"
            )

        return ModelPerformanceResponse(**result["metrics"])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.get("/results/{filename}", tags=["Results"])
async def get_results(filename: str):
    """
    Get prediction results for a specific file from MongoDB
    """
    try:
        result = await PredictionResult.get_by_filename(filename)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No results found for {filename}"
            )
        
        # Convert ObjectId to string
        result['_id'] = str(result['_id'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve results: {str(e)}"
        )


@app.get("/results",  tags=["Results"])
async def get_all_results():
    """
    Get all prediction results from MongoDB using the static method.
    """
    try:
        results = await PredictionResult.list_all()
        
        # Convert ObjectId to string for JSON serialization
        for r in results:
            r["_id"] = str(r["_id"])
        
        return {"count": len(results), "results": results}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve results: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)