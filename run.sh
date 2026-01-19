#!/bin/bash
# Quick start script for Voice AI with Deep Research

echo "🚀 Starting Voice AI with Deep Research..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Check environment variables
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "⚠️  Warning: DASHSCOPE_API_KEY not set"
    echo "   Please set it: export DASHSCOPE_API_KEY='your-key'"
fi

# Check if SearxNG is running
if ! curl -s http://127.0.0.1:38000 > /dev/null 2>&1; then
    echo "⚠️  Warning: SearxNG not accessible at http://127.0.0.1:38000"
    echo "   Start it with: docker run -d -p 38000:8080 searxng/searxng"
fi

# Run the voice agent
echo "🎤 Starting voice agent..."
cd voice-ai-by-cursor
# Run from voice-ai-by-cursor directory, script will add parent to path
python voice_ai_realtime.py
