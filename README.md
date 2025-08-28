# Artwork Object Detection Server

This fork removes SAM and Rembg processing and instead uses YOLO models trained on artwork to detect objects. Each model produces a single output image with bounding boxes and its name overlaid on the image.

## Model Weights

Place detector weights in `shared/models/` **before** starting the containers.
This directory is mounted at `/models` inside each container and the worker
automatically loads any YOLO `.pt` files it finds there.

Sample links:

- [yolov8n.pt](https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt)
- [rf_detr_r50.pth](https://github.com/lyuwenyu/rf-detr/releases/download/v0.1/rf_detr_r50.pth)
- [rt_detr_r50.pth](https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1/rt_detr_r50.pth)
- [dfine_r18.pth](https://github.com/lyuwenyu/D-FINE/releases/download/v0.1/dfine_r18.pth)

For line drawings, you may have better luck with models fine-tuned on sketch datasets; search sites like [Hugging Face](https://huggingface.co/models?search=line%20art%20yolo) for "line art" or "sketch" YOLO variants.

## Building

The Dockerfiles install the system libraries needed for `torchvision`.

```bash
# Server 1
docker build -t yolo-server1 -f Server1.Dockerfile .

# Server 2
docker build -t yolo-server2 -f Server2.Dockerfile .
```

## Running

```bash
# Server 1
docker run -it --rm -p 5050:5050 -v $(pwd)/shared:/mnt/shared -v $(pwd)/shared/models:/models yolo-server1

# Server 2
docker run -it --rm -v $(pwd)/shared:/mnt/shared -v $(pwd)/shared/models:/models yolo-server2
```
