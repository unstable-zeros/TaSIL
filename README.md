# Setup

 1. Install Docker

    See https://docs.docker.com/engine/install/ubuntu/

    Ensure that your user is added to the "docker" group

 2. Install the Nvidia Container Runtime with

    `sudo apt install nvidia-docker2`

    You will probably need to reboot

 3. Build the docker image

    From the root directory run

    `./docker/build.sh`

 4. Run the container with

    `./docker/run.sh`
    
    You should be dropped into a shell of `youruser@code>`

# Usage

Inside of the docker container use `train.py` to do training runs. For instance

 - `python train.py --environment gym/Ant-v3 --train\_trajs 20 --epochs 4500`
    
    For a OpenAI gym environment.
 - `python train.py --environment dummy2 --train\_trajs 20 --epochs 4500`
    
    For a synthetic exponentially stable system.

For a complete description of all available options, please see `train.py`
