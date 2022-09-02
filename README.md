# RandomWalk_NMR

# Dockerized

1. Setting up NVIDIA Container Toolkit
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

2. Install nvidia-docker2 package via apt:
```
sudo apt update && sudo apt install -y nvidia-docker2
```

3. Restart the Docker daemon to complete the installation:
```
sudo systemctl restart docker
```

4. Build Docker Image:
```
docker build -t rwnmr .
```

5. Run Docker Container:
```
docker run -it --rm --gpus all -v $(pwd)/db:/app/db -v $(pwd)/config:/app/config -v $(pwd)/input:/app/input rwnmr
```