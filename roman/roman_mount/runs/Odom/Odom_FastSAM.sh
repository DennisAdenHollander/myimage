#!/bin/bash
export ROMAN_DEMO_DATA=/workspace/roman/test_data
export ROMAN_WEIGHTS=/workspace/roman/weights

python3 /workspace/roman/demo/demo.py \
    -p /workspace/roman/params/Odom/FastSAM \
    -o /workspace/roman/roman_mount/Odom_results/FastSAM
    

