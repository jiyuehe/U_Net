NOTE: 
the minkowski engine docker container is Ubuntu 18  
3D data dimensions (t, depth, height, width)  

if docker is running, command line to kill all docker containers: 
docker stop $(docker ps -q)

tmux 
or 
tmux attach -t 0

docker run --gpus all -it \
    -v /home/j/Desktop/ssd/git:/workspace \
    -v /home/j/Desktop/hdd:/data \
    -p 8888:8888 \
    my_minkowski_image:latest /bin/bash

python3 U_Net/main_u_net.py

To exit docker, command line: exit
