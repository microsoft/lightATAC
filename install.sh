################ Uncommment this part to install conda env
# conda create -n lightATAC  python=3.9 -y
# conda activate lightATAC
################

################ Uncommment this part to install mujoco210
sudo apt-get install -y libglew-dev patchelf libosmesa6-dev
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
################
