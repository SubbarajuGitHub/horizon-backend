from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Optional
import os
from datetime import datetime
import boto3

from dotenv import load_dotenv
load_dotenv()

MONGODB_URL  =  os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Async MongoDB client (for FastAPI)
client: Optional[AsyncIOMotorClient] = None
database = None


async def connect_to_mongo():
    """Connect to MongoDB"""
    global client, database
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client[DATABASE_NAME]
    print(f"Connected to MongoDB: {DATABASE_NAME}")


async def close_mongo_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        print("Closed MongoDB connection")


def get_database():
    """Get database instance"""
    return database


# Collections
class Collections:
    """MongoDB collection names"""
    FILES = "files"
    PREDICTIONS = "predictions"
    MODELS = "models"


class FileMetadata:
    """Helper class to manage file metadata in MongoDB"""
    
    @staticmethod
    async def create(filename: str, original_name: str, file_size: int, rows: int, columns: int):
        """Store file metadata in MongoDB"""
        collection = database[Collections.FILES]
        
        file_doc = {
            "filename": filename,
            "original_name": original_name,
            "file_size": file_size,
            "rows": rows,
            "columns": columns,
            "uploaded_at": datetime.utcnow(),
            "status": "uploaded"
        }
        
        result = await collection.insert_one(file_doc)
        return str(result.inserted_id)
    
    @staticmethod
    async def get_by_filename(filename: str):
        """Get file metadata by filename"""
        collection = database[Collections.FILES]
        return await collection.find_one({"filename": filename})
    
    @staticmethod
    async def list_all():
        """List all uploaded files"""
        collection = database[Collections.FILES]
        cursor = collection.find({}).sort("uploaded_at", -1)
        files = await cursor.to_list(length=100)
        return files
    
    @staticmethod
    async def update_status(filename: str, status: str):
        """Update file processing status"""
        collection = database[Collections.FILES]
        await collection.update_one(
            {"filename": filename},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}}
        )


class PredictionResult:
    """Helper class to manage prediction results in MongoDB"""
    
    @staticmethod
    async def create(filename: str,original_filename: str, predictions: list, metrics: dict = None):
        """Store prediction results"""
        collection = database[Collections.PREDICTIONS]
        
        # Store only summary, not full predictions (to avoid size issues)
        result_doc = {
            "filename": filename,
            "original_filename": original_filename,
            "num_predictions": len(predictions),
            "metrics": metrics,
            "created_at": datetime.utcnow(),
            # Store only high-risk predictions (> 0.6 probability)
            "high_risk_count": len([
                p for p in predictions 
                if p.get('LogisticRegressionProb', 0) > 0.6 or p.get('RandomForestProb', 0) > 0.6
            ])
        }
        
        result = await collection.insert_one(result_doc)
        return str(result.inserted_id)
    
    @staticmethod
    async def get_by_filename(filename: str):
        """Get prediction results by filename"""
        collection = database[Collections.PREDICTIONS]
        return await collection.find_one({"filename": filename}, sort=[("created_at", -1)])
    
    @staticmethod
    async def list_all():
        """List all prediction results"""
        collection = database[Collections.PREDICTIONS]
        cursor = collection.find({}).sort("created_at", -1)
        results = await cursor.to_list(length=100)
        return results
