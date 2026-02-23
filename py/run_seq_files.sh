#!/bin/bash
# Loop over the following grid sizes:
# [256, 512, 1024, 1536, 2048]

if [[ $# -gt 1 ]]; then
  echo "Provided too many arguments"
  exit 1
fi

files=("seq_fft.py" "seq_laplace.py" "seq_stencil_explicit.py")
grids=(256 512 1024 1536 2048)

for file in "${files[@]}"; do
  echo "Executing $file"
  for num in "${grids[@]}"; do
    timeout 5m python "$file" "$num" "$num" | tee -a "./run_output_${file%.*}.log"
  done
done
