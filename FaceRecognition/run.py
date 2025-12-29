#!/usr/bin/env python3
"""
Quick start script for Face Recognition.
This script provides a simple interface to run the application.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main

if __name__ == "__main__":
    main()
