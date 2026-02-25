NOTE: 
the minkowski engine docker container is Ubuntu 18
2D data dimensions (t, height, width)  
3D data dimensions (t, depth, height, width)  

Use Minkowski Engine Docker
[Run my Docker container]
if docker is running, kill all docker containers: 
docker stop $(docker ps -q)

tmux 
or 
tmux attach -t 0

docker run --gpus all -it \
    -v /home/j/Desktop/ssd/git/HeartSim:/workspace \
    -v /home/j/Desktop/hdd:/data \
    -p 8888:8888 \
    my_minkowski_image:latest /bin/bash

python3 U_Net/main.py