#!/bin/bash

# Quick deployment script for Sentiment Analysis App
# This script handles port conflicts and provides multiple deployment options

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Sentiment Analysis App Deployment${NC}"
echo "=================================="

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to stop existing processes
cleanup_ports() {
    echo -e "${YELLOW}🧹 Cleaning up existing processes...${NC}"
    
    # Stop any existing Streamlit processes
    pkill -f "streamlit run" 2>/dev/null || true
    
    # Stop any existing Docker containers
    docker stop sentiment-analysis-app 2>/dev/null || true
    docker rm sentiment-analysis-app 2>/dev/null || true
}

# Function to run with Docker
run_docker() {
    local port=$1
    echo -e "${BLUE}🐳 Running with Docker on port $port...${NC}"
    
    # Build image if it doesn't exist
    if ! docker image inspect sentiment-analysis-app:latest >/dev/null 2>&1; then
        echo "Building Docker image..."
        docker build -t sentiment-analysis-app:latest .
    fi
    
    # Run container
    docker run -d -p $port:8501 --name sentiment-analysis-app sentiment-analysis-app:latest
    
    echo -e "${GREEN}✅ Docker container started!${NC}"
    echo -e "${GREEN}🌐 Access your app at: http://localhost:$port${NC}"
    echo -e "${YELLOW}📊 View logs with: docker logs sentiment-analysis-app${NC}"
    echo -e "${YELLOW}🛑 Stop with: docker stop sentiment-analysis-app${NC}"
}

# Function to run with Streamlit directly
run_streamlit() {
    local port=$1
    echo -e "${BLUE}📱 Running with Streamlit on port $port...${NC}"
    
    # Set environment variables to avoid inotify issues
    export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
    export STREAMLIT_SERVER_HEADLESS=true
    
    # Run Streamlit
    streamlit run dashboard/streamlit_app.py --server.port $port --server.address 0.0.0.0
}

# Main deployment logic
main() {
    # Find available port
    PORT=8501
    while check_port $PORT; do
        PORT=$((PORT + 1))
    done
    
    echo -e "${GREEN}✅ Found available port: $PORT${NC}"
    
    # Cleanup existing processes
    cleanup_ports
    
    # Show options
    echo ""
    echo "Deployment Options:"
    echo "1. Docker (Recommended for production)"
    echo "2. Streamlit (Direct, for development)"
    echo "3. Both (Docker + Streamlit on different ports)"
    
    read -p "Choose option (1-3): " choice
    
    case $choice in
        1)
            run_docker $PORT
            ;;
        2)
            run_streamlit $PORT
            ;;
        3)
            # Run Docker on found port
            run_docker $PORT
            # Find another port for Streamlit
            STREAMLIT_PORT=$((PORT + 1))
            while check_port $STREAMLIT_PORT; do
                STREAMLIT_PORT=$((STREAMLIT_PORT + 1))
            done
            echo -e "${BLUE}📱 Also starting Streamlit on port $STREAMLIT_PORT...${NC}"
            run_streamlit $STREAMLIT_PORT &
            echo -e "${GREEN}✅ Both services running!${NC}"
            echo -e "${GREEN}🐳 Docker: http://localhost:$PORT${NC}"
            echo -e "${GREEN}📱 Streamlit: http://localhost:$STREAMLIT_PORT${NC}"
            ;;
        *)
            echo -e "${YELLOW}Invalid option. Running Docker by default...${NC}"
            run_docker $PORT
            ;;
    esac
}

# Run main function
main "$@"
