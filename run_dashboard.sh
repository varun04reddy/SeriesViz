#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Add project root to PYTHONPATH to allow imports from 'src'
export PYTHONPATH=$(pwd)

# Run the Streamlit app
streamlit run src/app/main.py