#!/bin/bash
export ROMAN_DEMO_DATA=/workspace/roman/test_data
export ROMAN_WEIGHTS=/workspace/roman/weights

python3 /workspace/roman/demo/demo_maskdino.py \
    -p /workspace/roman/params/Odom/MaskDINO+Plane+Class+Conf \
    -o /workspace/roman/roman_mount/Odom_results/MaskDINO+Plane+Class+Conf
    

