#!/bin/bash
# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Download validation data and model
wget https://github.com/Xilinx/finn-examples/releases/download/kws/python_speech_preprocessing_all_validation_KWS_data.npz

#!/bin/bash

# Get path argument or use current directory
TARGET_DIR=${1:-$(pwd)}

FILE="$TARGET_DIR/onnx-models-kws.zip"
URL=https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a/onnx-models-kws.zip
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