#!/bin/bash
set -euo pipefail

export ROMAN_DEMO_DATA=/workspace/roman/test_data
export ROMAN_WEIGHTS=/workspace/roman/weights

# Odom
python3 /workspace/roman/demo/demo.py \
    -p /workspace/roman/params/Odom/FastSAM \
    -o /workspace/roman/roman_mount/Odom_results/FastSAM \
    --max-time 240
    
python3 /workspace/roman/demo/demo_maskdino.py \
    -p /workspace/roman/params/Odom/MaskDINO \
    -o /workspace/roman/roman_mount/Odom_results/MaskDINO \
    --max-time 240

python3 /workspace/roman/demo/demo_maskdino.py \
    -p /workspace/roman/params/Odom/MaskDINO+Plane \
    -o /workspace/roman/roman_mount/Odom_results/MaskDINO+Plane \
    --max-time 240  
    
python3 /workspace/roman/demo/demo_maskdino.py \
    -p /workspace/roman/params/Odom/MaskDINO+Plane+Class \
    -o /workspace/roman/roman_mount/Odom_results/MaskDINO+Plane+Class \
    --max-time 240    
    
python3 /workspace/roman/demo/demo_maskdino.py \
    -p /workspace/roman/params/Odom/MaskDINO+Plane+Class+Conf \
    -o /workspace/roman/roman_mount/Odom_results/MaskDINO+Plane+Class+Conf \
    --max-time 240  
    
# VIO
python3 /workspace/roman/demo/demo.py \
    -p /workspace/roman/params/VIO/FastSAM \
    -o /workspace/roman/roman_mount/VIO_results/FastSAM \
    --max-time 240 
    
python3 /workspace/roman/demo/demo_maskdino.py \
    -p /workspace/roman/params/VIO/MaskDINO \
    -o /workspace/roman/roman_mount/VIO_results/MaskDINO 
    --max-time 240    
    
python3 /workspace/roman/demo/demo_maskdino.py \
    -p /workspace/roman/params/VIO/MaskDINO+Plane \
    -o /workspace/roman/roman_mount/VIO_results/MaskDINO+Plane \
    --max-time 240
    
python3 /workspace/roman/demo/demo_maskdino.py \
    -p /workspace/roman/params/VIO/MaskDINO+Plane+Class \
    -o /workspace/roman/roman_mount/VIO_results/MaskDINO+Plane+Class \
    --max-time 240    
    
python3 /workspace/roman/demo/demo_maskdino.py \
    -p /workspace/roman/params/VIO/MaskDINO+Plane+Class+Conf \
    -o /workspace/roman/roman_mount/VIO_results/MaskDINO+Plane+Class+Conf \
    --max-time 240    

         
