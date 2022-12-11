#!/bin/bash
#SBATCH -p middle
#SBATCH -J DET_AFP2
#SBATCH -o %j-%x.log
#SBATCH -e %j-%x.log
#SBATCH -t 120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4


IMG="bcm-head001:5000/mmdet:latest"
WORK_DIR="/home/takashi.shibata.tc/050_titech/accv/DET_AFP"
WORK_DIR2="/home/takashi.shibata.tc/050_titech/mmdetection"
BASH="./tools/Faster/dist_faster_Prop_FT2.sh"

### << Main Job >> ###
# Use Docker
module load docker

# Pull Docker images to node local
slurm-docker pull ${IMG}

# Run Docker container with GPUs
slurm-docker run --nv \
        --shm-size=60g \
        --rm \
        --ipc=host \
        -v ${WORK_DIR}:/accv/DET_AFP \
        -v ${WORK_DIR2}:/mmdetection \
        -w /accv/DET_AFP \
        ${IMG} \
        bash ${BASH}               