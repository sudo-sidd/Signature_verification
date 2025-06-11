#!/bin/bash

# Signature Verification Portal Startup Script

echo "🚀 Starting Signature Verification Portal..."
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p signatures

# Start the server
echo "🖥️  Starting FastAPI server..."
echo "==============================================="
echo "🌐 Open your browser and go to: http://localhost:8000"
echo "🛑 Press Ctrl+C to stop the server"
echo "==============================================="

python main.py
