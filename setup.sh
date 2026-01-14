#!/bin/bash

# Synthetic Data Generation Platform - Setup Script
# This script helps set up the development environment

set -e  # Exit on any error

echo "🚀 Setting up Synthetic Data Generation Platform..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Start Docker Desktop if not running
echo "🐳 Checking Docker status..."
if ! docker info &> /dev/null; then
    echo "� Docker is not running. Checking if Docker Desktop is installed..."
    
    # Check if Docker Desktop is installed
    if [ -d "/Applications/Docker.app" ] || [ -d "$HOME/Applications/Docker.app" ]; then
        echo "✅ Docker Desktop found. Starting it..."
        if [ -d "/Applications/Docker.app" ]; then
            open "/Applications/Docker.app"
        else
            open "$HOME/Applications/Docker.app"
        fi
        
        echo "⏳ Waiting for Docker Desktop to start..."
        
        # Wait for Docker to be ready (max 90 seconds)
        for i in {1..18}; do
            if docker info &> /dev/null; then
                echo "✅ Docker Desktop is running"
                break
            fi
            echo "   Waiting... ($((i*5))s)"
            sleep 5
        done
        
        # Final check
        if ! docker info &> /dev/null; then
            echo "❌ Docker Desktop failed to start within 90 seconds."
            echo "Please check Docker Desktop manually and try again."
            exit 1
        fi
    else
        echo "❌ Docker Desktop is not installed."
        echo ""
        echo "To install Docker Desktop:"
        echo "1. Visit: https://www.docker.com/products/docker-desktop/"
        echo "2. Download Docker Desktop for Mac"
        echo "3. Install and start Docker Desktop"
        echo "4. Run this script again"
        echo ""
        echo "Alternatively, if Docker Desktop is installed elsewhere:"
        echo "- Start Docker Desktop manually from your Applications folder"
        echo "- Then run this script again"
        exit 1
    fi
else
    echo "✅ Docker is already running"
fi

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys and configuration"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data temp_uploads logs init-scripts

# Set execute permission on init script
chmod +x init-scripts/01-init.sh

# Check if required environment variables are set
echo "🔍 Checking environment variables..."

if [ -f .env ]; then
    source .env
    
    if [ -z "$GEMINI_API_KEY" ]; then
        echo "⚠️  GEMINI_API_KEY is not set in .env file"
    else
        echo "✅ GEMINI_API_KEY is configured"
    fi
    
    if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
        echo "⚠️  GOOGLE_CLOUD_PROJECT is not set in .env file"
    else
        echo "✅ GOOGLE_CLOUD_PROJECT is configured"
    fi
fi

# Build and start services
echo "🐳 Building and starting Docker containers..."
docker-compose build

echo "📊 Starting PostgreSQL database..."
docker-compose up -d postgres

echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 10

# Check if PostgreSQL is healthy
if docker-compose exec postgres pg_isready -U postgres; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready. Check the logs with: docker-compose logs postgres"
    exit 1
fi

echo "🚀 Starting the main application..."
docker-compose up -d app

# Wait a bit for the app to start
sleep 5

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📍 Access the application at: http://localhost:8501"
echo ""
echo "🔧 Useful commands:"
echo "  - View logs: docker-compose logs -f app"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo "  - Start with pgAdmin: docker-compose --profile admin up -d"
echo ""

if command -v open &> /dev/null; then
    echo "🌐 Opening application in browser..."
    sleep 2
    open http://localhost:8501
elif command -v xdg-open &> /dev/null; then
    echo "🌐 Opening application in browser..."
    sleep 2
    xdg-open http://localhost:8501
fi

echo "✨ Enjoy generating synthetic data!"