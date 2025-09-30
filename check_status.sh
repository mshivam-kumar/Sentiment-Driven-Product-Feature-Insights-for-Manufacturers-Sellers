#!/bin/bash

# Status check script for Sentiment Analysis App

echo "🔍 Checking Sentiment Analysis App Status"
echo "========================================"

# Check Docker container
echo ""
echo "🐳 Docker Container Status:"
if docker ps | grep -q sentiment-analysis-app; then
    echo "✅ Container is running"
    docker ps | grep sentiment-analysis-app
    echo ""
    echo "📊 Recent logs:"
    docker logs --tail 5 sentiment-analysis-app
else
    echo "❌ Container is not running"
fi

# Check Streamlit processes
echo ""
echo "📱 Streamlit Processes:"
if pgrep -f "streamlit run" > /dev/null; then
    echo "✅ Streamlit is running"
    ps aux | grep "streamlit run" | grep -v grep
else
    echo "❌ No Streamlit processes found"
fi

# Check ports
echo ""
echo "🌐 Port Status:"
for port in 8501 8502 8503 8504; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "✅ Port $port is in use"
        lsof -Pi :$port -sTCP:LISTEN
    else
        echo "❌ Port $port is free"
    fi
done

# Check Docker image
echo ""
echo "🐳 Docker Image:"
if docker image inspect sentiment-analysis-app:latest >/dev/null 2>&1; then
    echo "✅ Docker image exists"
    docker images | grep sentiment-analysis-app
else
    echo "❌ Docker image not found"
fi

echo ""
echo "🚀 Quick Commands:"
echo "  Start app: ./run_app.sh"
echo "  Stop Docker: docker stop sentiment-analysis-app"
echo "  Stop Streamlit: pkill -f 'streamlit run'"
echo "  View logs: docker logs sentiment-analysis-app"
