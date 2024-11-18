#!/bin/bash

# Check for Node.js and npm
if ! command -v node &> /dev/null
then
    echo "Node.js is not installed. Please install Node.js from https://nodejs.org/"
    exit
fi

if ! command -v npm &> /dev/null
then
    echo "npm is not installed. Please install npm from https://nodejs.org/"
    exit
fi

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
cd fringe-tracker
npm install

# Check for Python
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install Python3 from https://www.python.org/"
    exit
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Setup complete. You can now run the application."
