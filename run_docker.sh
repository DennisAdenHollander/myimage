xhost +local:root

docker run --rm -it \
  --gpus all \
  -v "$PWD/roman:/workspace/roman" \
  -v "$PWD/test_data:/workspace/roman/test_data" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  dennis

  

