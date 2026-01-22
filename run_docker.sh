xhost +local:root

docker run --rm -it \
  --gpus all \
  -v "$PWD/MaskDINO:/workspace/MaskDINO" \
  -v "$PWD/roman:/workspace/roman" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  dennis

  

