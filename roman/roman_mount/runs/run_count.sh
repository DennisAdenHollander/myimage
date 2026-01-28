#!/bin/bash
set -euo pipefail

export ROMAN_DEMO_DATA=/workspace/roman/test_data
export ROMAN_WEIGHTS=/workspace/roman/weights 
    
python3 /workspace/roman/demo/class_count.py \
    -p /workspace/roman/params/VIO/MaskDINO \
    -o /workspace/roman/roman_mount/Count_results \
    --max-time 240  
    
         
