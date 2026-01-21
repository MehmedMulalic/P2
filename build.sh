#!/bin/bash
unset CPLUS_INCLUDE_PATH

if [ -z "$1" ]; then
    echo "Usage: ./build.sh <filename> <compiler_flags>"
    exit 1
fi

FILE="$1"
OUT="${FILE%.*}"

if [ -n "$2" ]; then
    FLAG="$2"
    nvcc -ccbin /usr/bin/g++ "$FILE" -o "$OUT" -$FLAG #-Wno-deprecated-gpu-targets
else
    nvcc -ccbin /usr/bin/g++ "$FILE" -o "$OUT" #-Wno-deprecated-gpu-targets
fi