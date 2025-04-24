#!/bin/bash

# Get path argument or use current directory
TARGET_DIR=${1:-$(pwd)}

FILE="$TARGET_DIR/onnx-models-gtsrb.zip"
URL=https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a/onnx-models-gtsrb.zip
EXTRACTED_FILE="$TARGET_DIR/cnv_1w1a_gtsrb.onnx"  # Change this if another file should be checked

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Check if the file already exists
if [ -f "$FILE" ]; then
    echo "$FILE already exists. Skipping download."
else
    echo "Downloading $FILE to $TARGET_DIR..."
    wget -O "$FILE" "$URL"
fi

# Check if the extraction has already been done
if [ -f "$EXTRACTED_FILE" ]; then
    echo "Extraction already done. Skipping extraction."
else
    echo "Extracting $FILE to $TARGET_DIR..."
    unzip -o -j "$FILE" -d "$TARGET_DIR"
fi