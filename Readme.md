# Deep Reinforcement Learning : Applying Visual Attention for Minecraft Pig Chase

This example code trains an agent in Minecraft with reinforcement learning.<br>
In this example, the agent learns to chase and attack Pigs in Minecraft by reinforcement learning algorithms (PPO) **with only visual observation information (frame pixels, 120 x 160 x 3 channels)**.

This example uses [Project Malmo](https://www.microsoft.com/en-us/research/project/project-malmo/) (modded forge Minecraft for reinforcement learning) to run my agent on Minecraft for reinforcement learning.<br>
This example uses [Ray and RLlib](https://docs.ray.io/en/latest/rllib.html) for running RL agent.

![a trained RL agent](https://tsmatz.files.wordpress.com/2021/11/20211111_trained_agent.gif)

This provides instructions for running this example.

In this example, I have used **Ubuntu Server 20.04 LTS** in Microsoft Azure.<br>
This example (the following instruction) uses the real monitor, in which Minecraft UI will be shown. (Please configure extra display settings, if you run on a virtual monitor.)

## 1. Setup NVIDIA GPU software (drivers and libraries) ##

> Note : When you just run the trained agent without training, GPU-utilized machine won't necessarily be needed and you can then skip settings in this section. (Please make sure to install ```tensorflow==2.4.1``` on CPUs, instead of ```tensorflow-gpu==2.4.1```.)

In this example, a large model (multi-layered transformer) is used for training a RL agent.<br>
In order to speed up training, use GPU-utilized machine.

In this settings, we'll use CUDA version 11.0 and cuDNN versioin 8.0, because I will use TensorFlow 2.4.1.<br>
You should install the correct version of NVIDIA libraries, corresponding with TensorFlow version. (See [here](https://www.tensorflow.org/install/source#gpu) for details about compatible drivers in TensorFlow.)

First of all, install ```gcc``` and ```make``` tools for building utilities.

```
sudo apt-get update
sudo apt-get install build-essential
# # or install individual packages as follows
# sudo apt install -y gcc
# sudo apt-get install -y make
```

Install CUDA driver by running the following command. (After installation, make sure to be correctly installed by running ```nvidia-smi``` command.)

```
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
sudo sh cuda_11.0.2_450.51.05_linux.run
```

Install cuDNN libraries.<br>
To install cuDNN, download the corresponding version of packages from [NVIDIA developer site](https://developer.nvidia.com/cudnn) and install these packages. (Here I have downloaded and used version 8.0.5.)

```
sudo dpkg -i libcudnn8_8.0.5.39-1+cuda11.0_amd64.deb
sudo dpkg -i libcudnn8-dev_8.0.5.39-1+cuda11.0_amd64.deb
sudo dpkg -i libcudnn8-samples_8.0.5.39-1+cuda11.0_amd64.deb
```

## 2. Download and build Malmo ##

To install Malmo, you can use pre-built binary or build Malmo from source code.<br>
Here we download source code and build Malmo in Ubuntu 20.04.

The following is the entire installation script, but see [here](https://github.com/tsmatz/minecraft-rl-example) for details about steps to compile Malmo in Ubuntu 20.04.

```
# install python 3.6
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6

# configure to make python3 command to use python3.6
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --config python3

# install required components
sudo apt-get install \
  build-essential \
  libpython3.6-dev \
  openjdk-8-jdk \
  swig \
  doxygen \
  xsltproc \
  ffmpeg \
  python-tk \
  python-imaging-tk \
  zlib1g-dev

# set environment for Java
echo -e "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> ~/.bashrc
source ~/.bashrc

# update certificates
sudo update-ca-certificates -f

# download and build cmake
mkdir ~/cmake
cd ~/cmake
wget https://cmake.org/files/v3.11/cmake-3.11.0.tar.gz
tar xvf cmake-3.11.0.tar.gz
cd cmake-3.11.0
./bootstrap
make -j4
sudo make install
cd

# download and build boost
mkdir ~/boost
cd ~/boost
wget http://sourceforge.net/projects/boost/files/boost/1.66.0/boost_1_66_0.tar.gz
tar xvf boost_1_66_0.tar.gz
cd boost_1_66_0
./bootstrap.sh --with-python=/usr/bin/python3.6 --prefix=.
./b2 link=static cxxflags=-fPIC install
cd

# download and install Malmo
git clone https://github.com/Microsoft/malmo.git ~/MalmoPlatform
wget https://raw.githubusercontent.com/bitfehler/xs3p/1b71310dd1e8b9e4087cf6120856c5f701bd336b/xs3p.xsl -P ~/MalmoPlatform/Schemas
echo -e "export MALMO_XSD_PATH=$PWD/MalmoPlatform/Schemas" >> ~/.bashrc
source ~/.bashrc
cd ~/MalmoPlatform
mkdir build
cd build
cmake -DBoost_INCLUDE_DIR=/home/$USER/boost/boost_1_66_0/include -DBOOST_PYTHON_NAME=python3 -DCMAKE_BUILD_TYPE=Release ..
make install
cd

# after installation, configure to make python3 command to use python3.8 (default in Ubuntu 20.04)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --config python3
```

## 3. Install required packages ##

Install required packages with dependencies (such as, TensorFlow, Ray framework with RLlib, etc) as follows.<br>
In this example, I have used TensorFlow for RLlib backend, but you can also use PyTorch for running RLlib.

First set up PIP in python3.6.

```
sudo apt-get update
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip
sudo apt-get install python3.6-distutils
```

Now let's install python packages in python 3.6.

```
python3.6 -m pip install \
  gym==0.21.0 \
  lxml \
  numpy \
  pillow \
  tensorflow-gpu==2.4.1 \
  gpustat==0.6.0 \
  ray[default]==1.6.0 \
  dm-tree==0.1.7 \
  ray[rllib]==1.6.0 \
  ray[tune]==1.6.0 \
  attrs==19.1.0 \
  pandas
```

## 4. Configure desktop environment ##

Malmo is built on the modded Minecraft.<br>
It then needs monitor-attached environment, and here I configure X remote desktop environment and RDP service as follows.

```
sudo apt-get update
# while installation, select gdm3 for default display manager
sudo apt-get -y install xfce4
sudo apt-get -y install xrdp
sudo systemctl enable xrdp
echo xfce4-session >~/.xsession
sudo service xrdp restart
```

> Note : Run ```echo xfce4-session >~/.xsession``` for all users who runs the program.

Allow (Open) inbound port 3389 (which is default RDP port's number) in network settings to enable your client to connect to your server.

> Note : When you want to join into the same game with your own Minecraft client remotely, please open Minecraft port 25565 too.

## 5. Clone this repository ##

Clone this repository.

```
git clone https://github.com/tsmatz/minecraft-rl-pigchase-attention
cd minecraft-rl-pigchase-attention
```

Expand ```flat_world.zip``` (world data) in repository folder.

```
unzip flat_world.zip -d flat_world
```

Copy the generated python package ```~/MalmoPlatform/build/install/Python_Examples/MalmoPython.so``` in current folder as follows.

```
cp ~/MalmoPlatform/build/install/Python_Examples/MalmoPython.so .
```

## 6. Train an agent (Deep Reinforcement Learning) ##

Let's start training.

Login Ubuntu using remote desktop client and run the following commands **on monitor-attached shell** (such as, LXTerminal), because it will need to launch Minecraft UI.<br>
Now, run the training script (```train.py```) as follows.

```
cd minecraft-rl-pigchase-attention
python3.6 train.py --num_gpus 1
```

> Note : For the first time to run, all dependencies for building Minecraft (including Project Malmo's mod) are built and installed, and it will then take a while to start. Please be patient to wait.<br>
> When you have troubles (errors) for downloading resources in minecraft compilation, please download [here](https://1drv.ms/u/s!AuopXnMb-AqcgdZkjmtSVg3VQL5TEQ?e=w4M4r7) and run the following command to use successful gradle cache.<br>
> ```mv ~/.gradle/caches/minecraft ~/.gradle/caches/minecraft-org```<br>
> ```unzip gradle_caches_minecraft.zip -d ~/.gradle/caches```

> Note : For other troubleshooting in building Minecraft or using monitor in Minecraft, see [here](https://github.com/tsmatz/minecraft-rl-example).

When you start the training code (```train.py```), you will see the agent's view in 160 x 120 Minecraft's screen. This frame pixels are then used by agent to train.

See "[Run Reinforcement Learning on Ray Cluster](https://tsmatz.wordpress.com/2021/10/08/rllib-reinforcement-learning-multiple-machines-ray-cluster/)", when you run on multiple workers in Ray cluster to speed up.

## 7. Run pre-trained agent

This repository also includes pre-trained checkpoint (```checkpoint/checkpoint-XXX```) in this repo.<br>
You can then check the result soon.

Run the following command to run the pre-trained agent, also **on monitor-attached shell** (such as, LXTerminal).

```
cd minecraft-rl-pigchase-attention
python3.6 run_agent.py
```

If you have your own trained checkpoint, you can also run your trained agent as follows.

```
python3.6 run_agent.py --checkpoint_file YOUR_OWN_CHECKPOINT_FILE_PATH
```
