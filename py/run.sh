#!/bin/bash
# 256, 512, 1024, 1536, 1792, 1920, 1984

USAGE="(./run.sh file_name [arg1] [arg2])"

if [[ $# -gt 2 ]]; then
  echo "Provided too many arguments $USAGE"
  exit 1
fi

if [[ -z "$1" ]]; then
  echo "Bad usage $USAGE"
  exit 1
fi

python "$1" "$2" "$2"