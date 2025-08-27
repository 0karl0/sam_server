
wget -nc https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-DIS-epoch_590.onnx -O shared/models/birefnet-dis.onnx

wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O shared/models/vit_l.pth
docker build -t sam-server1 -f Server1.Dockerfile .

docker run -it --rm -p 5050:5050 -v $(pwd)/shared:/mnt/shared -e RUNPOD_API_KEY="your_api_key_here" -e GPU_POD_ID="your_gpu_pod_id_here" sam-server1

docker build -t sam-server2 -f Server2.Dockerfile .

docker run -it --rm -v $(pwd)/shared:/mnt/shared sam-server2
