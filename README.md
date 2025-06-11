# Signature Verification Portal

A web-based application that uses machine learning to create and verify digital signatures. The portal allows users to draw signatures directly in the browser and verify their authenticity using AI-powered image recognition.

## Features

- **Canvas-Based Drawing**: Draw signatures directly in the browser with mouse/touch support
- **Signature Verification**: Verify signatures against stored database using ML
- **Simplified User Management**: Streamlined interface showing only names and signature status
- **AI-Powered**: Uses MobileNetV2 for feature extraction and cosine similarity for matching
- **Mobile-Friendly**: Touch-enabled signature drawing for mobile devices
- **Secure Storage**: Images stored locally with SQLite database for metadata

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework for APIs
- **TensorFlow**: Machine learning library with MobileNetV2 model
- **SQLite**: Lightweight database for user and signature data
- **Pillow**: Python Imaging Library for image processing
- **uvicorn**: ASGI server for running FastAPI

### Frontend
- **HTML5 Canvas**: For drawing signatures directly in browser
- **Vanilla JavaScript**: Modern ES6+ features with SignatureCanvas class
- **CSS3**: Responsive design with flexbox and modern styling
- **Touch Events**: Mobile-friendly signature drawing support

### Machine Learning
- **MobileNetV2**: Pre-trained CNN for feature extraction
- **Cosine Similarity**: Algorithm for comparing signature features
- **Image Preprocessing**: Resize, normalize, and prepare images for ML

## Project Structure

```
Signature_verification/
├── main.py              # FastAPI backend server
├── index.html           # Standalone web application with drawing canvas
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── signatures/         # Directory for signature images (created automatically)
└── signatures.db       # SQLite database (created automatically)
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- At least 8GB RAM (for TensorFlow)

### Step 1: Clone or Download
```bash
cd /path/to/your/projects
# If you have the files, navigate to the directory
cd Signature_verification
```

### Step 2: Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Start the Backend Server
```bash
python main.py
```

The server will start on `http://localhost:8000` and will automatically:
- Initialize the SQLite database
- Create the signatures directory
- Download and load the MobileNetV2 model (this may take a few minutes on first run)

### Step 4: Access the Application
Open your web browser and navigate to:
```
http://localhost:8000
```

## Usage Guide

### Creating a Signature

1. Click **"Draw Signature"** on the home page
2. Enter your full name (email is auto-generated)
3. Draw your signature using the canvas with mouse or finger
4. Use the **Clear** button to redraw if needed
5. Click **"Save Signature"** to store your signature
6. The system will process and store your signature with ML features

### Verifying a Signature

1. Click **"Verify Signature"** on the home page
2. Enter the name of the registered user
3. Draw the signature to verify using the canvas
4. Click **"Verify Signature"**
5. View the results with confidence score and verification status

### Understanding Results

- **Confidence Score**: Percentage indicating how similar the signatures are
- **Threshold**: 85% - signatures above this are considered verified
- **Status**: ✅ VERIFIED (≥85%) or ❌ NOT VERIFIED (<85%)

## API Endpoints

### POST /create-signature
Create a new signature entry
- **Form Data**: `name`, `signature` (base64 canvas data)
- **Response**: Success/error message with signature ID

### POST /verify-signature
Verify a signature against stored database
- **Form Data**: `name`, `signature` (base64 canvas data)
- **Response**: Verification result with confidence score

### GET /users
Get list of all registered users
- **Response**: Array of users with name and signature status only

## Technical Details

### Machine Learning Pipeline

1. **Image Preprocessing**:
   - Convert canvas data from base64 to image format
   - Convert to RGB format and resize to 224x224 pixels
   - Normalize pixel values using MobileNetV2 preprocessing

2. **Feature Extraction**:
   - Use pre-trained MobileNetV2 (without top layers)
   - Extract 1280-dimensional feature vectors
   - Global average pooling for consistent output size

3. **Similarity Calculation**:
   - Compute cosine similarity between feature vectors
   - Convert to percentage confidence score
   - Apply 85% threshold for verification

### Security Features

- Canvas data validation and processing
- Input sanitization for form data
- Secure file storage with hashed names
- CORS middleware for cross-origin requests
- Auto-generated dummy emails for simplified user management

### Database Schema

**Users Table**:
- id (Primary Key)
- name (Text)
- email (Text, auto-generated)
- created_at (Timestamp)

**Signatures Table**:
- id (Primary Key)
- user_id (Foreign Key)
- signature_hash (Text)
- features (Blob - stored ML features)
- image_path (Text)
- created_at (Timestamp)

## Customization Options

### Adjusting Verification Threshold
In `main.py`, modify the threshold value:
```python
threshold = 0.85  # Change to 0.8 for more lenient matching
```

### Changing Image Processing
Modify the `preprocess_image` function to adjust:
- Image size
- Color channels
- Normalization methods

### UI Customization
Edit `index.html` to:
- Change colors and styling in the CSS section
- Modify canvas drawing settings
- Add new features to the SignatureCanvas class
- Customize layout and components


## Development and Learning

### Key Learning Concepts

1. **Machine Learning Integration**:
   - Pre-trained model usage
   - Feature extraction vs full training
   - Similarity metrics and thresholds

2. **Web Development**:
   - REST API design with FastAPI
   - HTML5 Canvas manipulation and drawing
   - Frontend-backend communication with FormData

3. **Canvas and Drawing**:
   - HTML5 Canvas API usage
   - Mouse and touch event handling
   - Base64 image data conversion
   - Responsive drawing interfaces


## License

This project is created for educational purposes. Feel free to modify and extend it for your learning and development needs.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the API endpoints and error messages
3. Examine browser console for frontend errors
4. Check server logs for backend issues

---


This project combines web development, machine learning, and image processing to create a practical application that you can understand, modify, and extend as you learn.
