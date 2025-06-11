import os
import sqlite3
import hashlib
import numpy as np
from datetime import datetime
from typing import Optional, List
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import base64

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Signature Verification Portal", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("signatures", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize MobileNetV2 model for feature extraction
print("Loading MobileNetV2 model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("Model loaded successfully!")

# Database setup
def init_database():
    conn = sqlite3.connect('signatures.db')
    cursor = conn.cursor()
    
    # Drop and recreate tables to remove UNIQUE constraint on email
    cursor.execute('DROP TABLE IF EXISTS signatures')
    cursor.execute('DROP TABLE IF EXISTS users')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            signature_hash TEXT NOT NULL,
            features BLOB NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_database()

# Image preprocessing functions
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for feature extraction"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 224x224 (MobileNetV2 input size)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Expand dimensions for batch processing
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def extract_features(image_array: np.ndarray) -> np.ndarray:
    """Extract features using MobileNetV2"""
    try:
        features = base_model.predict(image_array, verbose=0)
        return features.flatten()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")

def calculate_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """Calculate cosine similarity between two feature vectors"""
    similarity = cosine_similarity([features1], [features2])[0][0]
    return float(similarity)

def save_signature_image(image_bytes: bytes, signature_hash: str) -> str:
    """Save signature image to disk"""
    image_path = f"signatures/{signature_hash}.png"
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.save(image_path, "PNG")
        return image_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/create-signature")
async def create_signature(
    name: str = Form(...),
    email: str = Form(...),
    signature: UploadFile = File(...)
):
    """Create a new signature entry"""
    
    # Validate file type
    if signature.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are allowed")
    
    # Read and validate file size (5MB limit)
    image_bytes = await signature.read()
    if len(image_bytes) > 5 * 1024 * 1024:  # 5MB
        raise HTTPException(status_code=400, detail="File size must be less than 5MB")
    
    try:
        # Generate hash for the signature
        signature_hash = hashlib.md5(image_bytes + name.encode() + email.encode()).hexdigest()
        
        # Preprocess image and extract features
        image_array = preprocess_image(image_bytes)
        features = extract_features(image_array)
        
        # Save image
        image_path = save_signature_image(image_bytes, signature_hash)
        
        # Save to database
        conn = sqlite3.connect('signatures.db')
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if user:
            user_id = user[0]
            # Update user name if different
            cursor.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))
        else:
            # Create new user
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
            user_id = cursor.lastrowid
        
        # Check if signature already exists for this user
        cursor.execute("SELECT id FROM signatures WHERE user_id = ?", (user_id,))
        existing_signature = cursor.fetchone()
        
        if existing_signature:
            # Update existing signature
            cursor.execute("""
                UPDATE signatures 
                SET signature_hash = ?, features = ?, image_path = ?, created_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (signature_hash, features.tobytes(), image_path, user_id))
            message = "Signature updated successfully"
        else:
            # Create new signature
            cursor.execute("""
                INSERT INTO signatures (user_id, signature_hash, features, image_path)
                VALUES (?, ?, ?, ?)
            """, (user_id, signature_hash, features.tobytes(), image_path))
            message = "Signature created successfully"
        
        conn.commit()
        conn.close()
        
        return JSONResponse(content={
            "success": True,
            "message": message,
            "signature_id": signature_hash
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating signature: {str(e)}")

@app.post("/verify-signature")
async def verify_signature(
    email: str = Form(...),
    signature: UploadFile = File(...)
):
    """Verify a signature against stored signatures"""
    
    # Validate file type
    if signature.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are allowed")
    
    # Read and validate file size
    image_bytes = await signature.read()
    if len(image_bytes) > 5 * 1024 * 1024:  # 5MB
        raise HTTPException(status_code=400, detail="File size must be less than 5MB")
    
    try:
        # Preprocess image and extract features
        image_array = preprocess_image(image_bytes)
        test_features = extract_features(image_array)
        
        # Get stored signature for user
        conn = sqlite3.connect('signatures.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.features, u.name, s.created_at
            FROM signatures s
            JOIN users u ON s.user_id = u.id
            WHERE u.email = ?
        """, (email,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return JSONResponse(content={
                "success": False,
                "message": "No signature found for this email address",
                "confidence": 0
            })
        
        # Extract stored features
        stored_features = np.frombuffer(result[0], dtype=np.float32)
        user_name = result[1]
        created_at = result[2]
        
        # Calculate similarity
        similarity = calculate_similarity(test_features, stored_features)
        confidence = similarity * 100
        
        # Determine if signatures match (threshold: 85%)
        threshold = 0.85
        is_match = similarity >= threshold
        
        return JSONResponse(content={
            "success": True,
            "is_match": is_match,
            "confidence": round(confidence, 2),
            "user_name": user_name,
            "signature_date": created_at,
            "threshold": threshold * 100,
            "message": f"Signature {'verified' if is_match else 'not verified'}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying signature: {str(e)}")

@app.get("/users")
async def get_users():
    """Get list of all users with signatures"""
    try:
        conn = sqlite3.connect('signatures.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.name
            FROM users u
            JOIN signatures s ON u.id = s.user_id
            ORDER BY u.created_at DESC
        """)
        
        users = []
        for row in cursor.fetchall():
            users.append({
                "name": row[0]
            })
        
        conn.close()
        
        return JSONResponse(content={
            "success": True,
            "users": users,
            "count": len(users)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")

if __name__ == "__main__":
    print("Starting Signature Verification Portal...")
    print("Make sure to create index.html in the same directory")
    uvicorn.run(app, host="0.0.0.0", port=8000)
