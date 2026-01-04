#!/bin/bash

# Activate conda environment if needed
# source activate face_recognition

# Run the FastAPI app
python -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload
