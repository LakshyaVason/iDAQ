#!/usr/bin/env bash
#
# Quick start script for iDAQ Diagnostics
# 
# Usage:
#   chmod +x quickstart.sh
#   ./quickstart.sh
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║           iDAQ Diagnostics - Quick Start                  ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Install Python 3.10 or higher"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python --version)${NC}"

# Check .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file not found${NC}"
    if [ -f ".env.example" ]; then
        echo "Creating .env from template..."
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env file${NC}"
        echo -e "${YELLOW}⚠ Please edit .env and add your API keys${NC}"
        echo ""
        echo "Required keys:"
        echo "  - OPENAI_API_KEY"
        echo "  - Firebase configuration"
        echo ""
        read -p "Press Enter after you've edited .env..."
    else
        echo -e "${RED}✗ .env.example not found${NC}"
        exit 1
    fi
fi

# Check if requirements are installed
if ! python -c "import fastapi" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Run diagnostics
echo ""
echo "Running system diagnostics..."
python diagnose.py

# Check if diagnostics passed
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Diagnostics failed${NC}"
    echo "Please fix the issues above before starting the server"
    exit 1
fi

# Ask to start server
echo ""
echo -e "${GREEN}✓ All checks passed!${NC}"
echo ""
read -p "Start the server now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}Starting iDAQ server...${NC}"
    echo ""
    echo "Access the application at:"
    echo -e "${GREEN}  http://localhost:8000${NC}"
    echo ""
    echo "Admin panel:"
    echo -e "${GREEN}  http://localhost:8000/admin${NC}"
    echo "  (username: pqlab, password: PQ2025!)"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""
    
    # Start server
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
else
    echo ""
    echo "To start the server manually, run:"
    echo -e "${GREEN}  source .venv/bin/activate${NC}"
    echo -e "${GREEN}  uvicorn main:app --reload --host 0.0.0.0 --port 8000${NC}"
fi