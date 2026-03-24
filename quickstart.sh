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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║           iDAQ Diagnostics - Quick Start                  ║"
echo "║           C2000 UART + AI-Powered Diagnostics             ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python (supports both python3 and python)
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}✗ Python not found${NC}"
    echo "Install Python 3.10 or higher"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $($PYTHON_CMD --version)${NC}"

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
        echo "  - OPENAI_API_KEY (required)"
        echo "  - C2000_UART_PORT (default: /dev/ttyTHS1)"
        echo "  - C2000_UART_BAUD (default: 115200)"
        echo "  - Firebase configuration (optional)"
        echo ""
        read -p "Press Enter after you've edited .env..."
    else
        echo -e "${RED}✗ .env.example not found${NC}"
        exit 1
    fi
fi

# Check OpenAI API key
if ! grep -q "^OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    echo -e "${YELLOW}⚠ OPENAI_API_KEY not configured in .env${NC}"
    echo "AI chat features will not work without this key"
fi

# Check if requirements are installed
echo ""
echo "Checking Python dependencies..."
if ! $PYTHON_CMD -c "import fastapi" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Check for C2000 UART device
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  C2000 UART Hardware Check${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Get UART port from .env or use default
UART_PORT=$(grep "^C2000_UART_PORT=" .env 2>/dev/null | cut -d= -f2)
if [ -z "$UART_PORT" ]; then
    UART_PORT="/dev/ttyTHS1"
fi

echo "Checking for UART device: $UART_PORT"

if ls "$UART_PORT" &>/dev/null || sudo ls "$UART_PORT" &>/dev/null; then
    echo -e "${GREEN}✓ UART device found: $UART_PORT${NC}"
    
    # Check permissions
    if [ -r "$UART_PORT" ] && [ -w "$UART_PORT" ]; then
        echo -e "${GREEN}✓ UART device is readable and writable${NC}"
    else
        echo -e "${YELLOW}⚠ UART device exists but insufficient permissions${NC}"
        echo ""
        echo "Run one of these commands to fix:"
        echo -e "${CYAN}  sudo chmod 666 $UART_PORT${NC}  (temporary)"
        echo -e "${CYAN}  sudo usermod -a -G dialout \$USER${NC}  (permanent, requires logout)"
        echo ""
        read -p "Try to fix permissions now? (requires sudo) (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo chmod 666 $UART_PORT
            echo -e "${GREEN}✓ Permissions updated${NC}"
        fi
    fi
    
    # Test if data is streaming
    echo ""
    echo "Testing UART data stream (5 second timeout)..."
    stty -F $UART_PORT 921600 2>/dev/null
    if timeout 5 dd if=$UART_PORT of=/tmp/uart_test.bin bs=1 count=120 2>/dev/null; then
        if xxd /tmp/uart_test.bin 2>/dev/null | grep -q "a5 5a"; then
            echo -e "${GREEN}✓ C2000 binary packet stream detected!${NC}"
            C2000_CONNECTED=true
        else
            echo -e "${YELLOW}⚠ Data received but binary sync bytes (0xA5 0x5A) not found${NC}"
            echo "  C2000 may still be running old firmware — flash the new binary firmware"
            C2000_CONNECTED=false
        fi
        rm -f /tmp/uart_test.bin
    else
        echo -e "${YELLOW}⚠ No data received from $UART_PORT${NC}"
        echo "Possible issues:"
        echo "  - C2000 not powered on"
        echo "  - Wrong UART port (check wiring)"
        echo "  - C2000 firmware not running"
        C2000_CONNECTED=false
    fi
else
    echo -e "${YELLOW}⚠ UART device not found: $UART_PORT${NC}"
    echo ""
    echo "Possible UART devices on this system:"
    ls -l /dev/tty{USB,ACM,THS}* 2>/dev/null | sed 's/^/  /' || echo "  (none found)"
    echo ""
    echo "If using USB-to-UART adapter, update .env:"
    echo "  C2000_UART_PORT=/dev/ttyUSB0"
    echo ""
    C2000_CONNECTED=false
fi
# Ask if user wants to start session logger
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  DSP Session Logger${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -f "dsp_session_logger.py" ]; then
    echo "Session logger found. This captures raw UART data to timestamped CSV files"
    echo "for debugging gain equations."
    echo ""
    read -p "Start session logger alongside the server? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        UART_BAUD=$(grep "^C2000_UART_BAUD=" .env 2>/dev/null | cut -d= -f2)
        if [ -z "$UART_BAUD" ]; then
            UART_BAUD="460800"
        fi
        echo -e "${GREEN}✓ Session logger will start with port=$UART_PORT baud=$UART_BAUD${NC}"
        START_LOGGER=true
    else
        START_LOGGER=false
    fi
else
    echo -e "${YELLOW}⚠ dsp_session_logger.py not found - skipping${NC}"
    START_LOGGER=false
fi
# Check for CSV fallback data
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Data Source Check${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ "$C2000_CONNECTED" = true ]; then
    echo -e "${GREEN}✓ Primary data source: C2000 UART (live hardware)${NC}"
elif [ -f "VinIinMOSFETVdsSCRVds_240_ALL.csv" ]; then
    echo -e "${YELLOW}⚠ Primary data source: CSV file (recorded data)${NC}"
    CSV_LINES=$(wc -l < VinIinMOSFETVdsSCRVds_240_ALL.csv)
    echo "  File: VinIinMOSFETVdsSCRVds_240_ALL.csv ($CSV_LINES lines)"
else
    echo -e "${YELLOW}⚠ Primary data source: Simulation (random data)${NC}"
    echo "  No C2000 connection and no CSV file found"
    echo ""
    echo "To use CSV playback mode, place your CSV file here:"
    echo "  VinIinMOSFETVdsSCRVds_240_ALL.csv"
fi

# Run diagnostics if script exists
if [ -f "diagnose.py" ]; then
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  System Diagnostics${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    $PYTHON_CMD diagnose.py

    if [ $? -ne 0 ]; then
        echo ""
        echo -e "${RED}✗ Diagnostics failed${NC}"
        echo "Please fix the issues above before starting the server"
        exit 1
    fi
fi

# Summary
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Startup Summary${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo ""
if [ "$C2000_CONNECTED" = true ]; then
    echo -e "${GREEN}✓ Ready to start with LIVE C2000 hardware data${NC}"
elif [ -f "VinIinMOSFETVdsSCRVds_240_ALL.csv" ]; then
    echo -e "${GREEN}✓ Ready to start with CSV playback mode${NC}"
else
    echo -e "${GREEN}✓ Ready to start in simulation mode${NC}"
fi

echo ""
echo "Server will automatically use the best available data source:"
echo "  1. C2000 UART (live hardware) → preferred"
echo "  2. CSV file (recorded data)   → fallback"
echo "  3. Simulation (random data)   → last resort"

# Ask to start server
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
    echo "User dashboard (main interface):"
    echo -e "${GREEN}  http://localhost:8000/user${NC}"
    echo ""
    echo "Admin panel (dataset management):"
    echo -e "${GREEN}  http://localhost:8000/admin${NC}"
    echo "  (username: pqlab, password: PQ2025!)"
    echo ""
    echo "API endpoints:"
    echo -e "${CYAN}  http://localhost:8000/health${NC}        - Health check"
    echo -e "${CYAN}  http://localhost:8000/data-source${NC}  - Active data source"
    echo -e "${CYAN}  http://localhost:8000/sensor-data${NC}  - Latest readings"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""
    
    if [ "$START_LOGGER" = true ]; then
        echo -e "${BLUE}Starting DSP session logger in background...${NC}"
        $PYTHON_CMD dsp_session_logger.py --port "$UART_PORT" --baud "$UART_BAUD" &
        LOGGER_PID=$!
        echo -e "${GREEN}✓ Logger running (PID $LOGGER_PID) → logs saved to dsp_logs/${NC}"
        echo ""
    fi

    # Start server
    $PYTHON_CMD -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

    if [ "$START_LOGGER" = true ] && kill -0 "$LOGGER_PID" 2>/dev/null; then
        echo ""
        echo -e "${YELLOW}Stopping session logger (PID $LOGGER_PID)...${NC}"
        kill "$LOGGER_PID"
    fi
    
else
    echo ""
    echo "To start the server manually, run:"
    echo -e "${GREEN}  uvicorn main:app --reload --host 0.0.0.0 --port 8000${NC}"
    echo ""
    echo "Or with production settings:"
    echo -e "${GREEN}  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4${NC}"
    echo ""
    echo "To test C2000 connection directly:"
    echo -e "${GREEN}  $PYTHON_CMD c2000_serial_reader.py${NC}"
fi