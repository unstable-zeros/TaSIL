#!/usr/bin/env bash
groupadd --gid $EXT_GID $EXT_USER
useradd -d /home/user -s /usr/bin/fish --uid $EXT_UID -g $EXT_USER $EXT_USER
# Fix some dumb mujoco stuff
chown -R $EXT_USER:$EXT_USER /usr/local/lib/python3.8/dist-packages/mujoco_py
cd /home/user/code
su $EXT_USER
