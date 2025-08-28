# Artwork Object Detection Server

This fork removes SAM and Rembg processing and instead uses detection models trained on artwork (YOLO, DETR, D-FINE) to detect objects. Each model produces a single output image with bounding boxes and its name overlaid on the image.

## Model Weights

Place any detection weights (`.pt`/`.pth`) in `shared/models` (mounted at
`/models` in the containers). Files with `detr` or `dfine` in their names are
treated as DETR or D‑FINE models; everything else loads through the YOLO
interface. The worker monitors this directory and loads new weights on the fly
so you can drop in additional models without restarting the container.

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
