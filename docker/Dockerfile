FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

## preesed tzdata, update package index, upgrade packages and install needed software
RUN truncate -s0 /tmp/preseed.cfg; \
    echo "tzdata tzdata/Areas select America" >> /tmp/preseed.cfg; \
    echo "tzdata tzdata/Zones/America select New_York" >> /tmp/preseed.cfg; \
    debconf-set-selections /tmp/preseed.cfg; \
    rm -f /etc/timezone /etc/localtime; \
    apt-get update; \
    apt-get upgrade;
    apt-get install -y tzdata && \
    apt-get install -y \
        build-essential wget curl \
        bash python3 python3-pip python3-dev python-is-python3 fish git neovim \
        libcairo2-dev pkg-config libcudnn8 gcc libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/*
# apt-get install -y software-properties-common; \
# add-apt-repository ppa:deadsnakes/ppa; \
# apt-get update; \
# Use python3.7 as python
# RUN ln -s /usr/bin/python3.7 /usr/bin/python
# # Install pip manually
# RUN curl https://bootstrap.pypa.io/get-pip.py -o /get-pip.py
# RUN python /get-pip.py
# RUN rm /get-pip.py
# RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN wget https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf -O /usr/local/bin/patchelf
RUN chmod +x /usr/local/bin/patchelf

# Install mujoco
RUN mkdir -p /opt/mujoco
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /opt/mujoco/mujoco210-linux-x86_64.tar.gz
RUN tar xf /opt/mujoco/mujoco210-linux-x86_64.tar.gz -C /opt/mujoco
RUN rm /opt/mujoco/mujoco210-linux-x86_64.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH /opt/mujoco/mujoco210
ENV LD_LIBRARY_PATH /opt/mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

COPY requirements.txt.tmp /requirements.txt
RUN pip3 install -r /requirements.txt -f https://storage.googleapis.com/jax-releases/jax_releases.html --extra-index-url https://download.pytorch.org/whl/cu113
# Trigger stupid mujoco build stuff so that the user doesn't have to rebuild
RUN python -c "from gym.envs.mujoco.mujoco_env import MujocoEnv"

RUN echo "root:docker" | chpasswd
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT /entrypoint.sh
