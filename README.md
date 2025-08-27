# Artwork Object Detection Server

This fork removes SAM and Rembg processing and instead uses YOLO models trained on artwork to detect objects. Each model produces a single output image with bounding boxes and its name overlaid on the image.

## Building

The Dockerfiles automatically download the required YOLOv8 model weights and install the system libraries needed for `torchvision`.

```bash
# Server 1
docker build -t yolo-server1 -f Server1.Dockerfile .

# Server 2
docker build -t yolo-server2 -f Server2.Dockerfile .
```

## Running

```bash
# Server 1
docker run -it --rm -p 5050:5050 -v $(pwd)/shared:/mnt/shared yolo-server1

# Server 2
docker run -it --rm -v $(pwd)/shared:/mnt/shared yolo-server2
```
