#!/bin/bash

# Google Drive file ID (ganti dengan ID file Anda)
FILE_ID="1CLyBPOm1L7kUHxczuvgcOqMv1vtIHA28"
MODEL_PATH="model.h5"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model already exists, skipping download"
    exit 0
fi

echo "📥 Downloading model from Google Drive..."

# Install gdown if not exists
pip install gdown

# Download the model
echo "🔄 Downloading with FILE_ID: $FILE_ID"
gdown --id $FILE_ID --output $MODEL_PATH

# Check if download was successful
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model downloaded successfully"
    ls -lh $MODEL_PATH
else
    echo "❌ Failed to download model"
    echo "🔍 Trying alternative download method..."
    
    # Alternative method using wget
    wget --no-check-certificate "https://drive.google.com/uc?export=download&id=$FILE_ID" -O $MODEL_PATH
    
    if [ -f "$MODEL_PATH" ]; then
        echo "✅ Model downloaded successfully with alternative method"
        ls -lh $MODEL_PATH
    else
        echo "❌ All download methods failed"
        exit 1
    fi
fi
