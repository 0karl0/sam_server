# Artwork Object Detection Server

This fork removes SAM and Rembg processing and instead uses YOLO models trained on artwork to detect objects. Each model produces a single output image with bounding boxes and its name overlaid on the image.

## Model Weights

Download the desired YOLOv8 weight files and place them in `shared/models/` before running the containers. The worker loads every `.pt` file in that folder automatically, so you can drop in additional models to experiment.

Common choices:

- [yolov8n.pt](https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt)
- [yolov8s.pt](https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8s.pt)
- [yolov8m.pt](https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8m.pt)
- [yolov8l.pt](https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8l.pt)

For line drawings, you may have better luck with models fine-tuned on sketch datasets; search sites like [Hugging Face](https://huggingface.co/models?search=line%20art%20yolo) for "line art" or "sketch" YOLO variants.

## Building

The Dockerfiles install the system libraries needed for `torchvision` but do **not** download any model weights.

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
docker run -it --rm -v $(pwd)/shared:/mnt/shared -v $(pwd)/shared/models:/models yolo-server2
```
