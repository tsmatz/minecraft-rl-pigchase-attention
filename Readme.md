# Deep Reinforcement Learning : Applying Visual Attention for Minecraft Pig Chase

This example code trains an agent in Minecraft with reinforcement learning.<br>
In this example, the agent learns to chase and attack Pigs in Minecraft by reinforcement learning algorithms (PPO) **with only visual observation information (frame pixels, 120 x 160 x 3 channels)**.

This example uses [Project Malmo](https://www.microsoft.com/en-us/research/project/project-malmo/) (modded forge Minecraft for reinforcement learning) to run my agent on Minecraft for reinforcement learning.<br>
This example uses [Ray and RLlib](https://docs.ray.io/en/latest/rllib.html) for running RL agent.

![a trained RL agent](https://tsmatz.files.wordpress.com/2021/11/20211111_trained_agent.gif)

This provides instructions for running this example.

In this example, I have used **Ubuntu Server 18.04 LTS** in Microsoft Azure.<br>
This example (the following instruction) uses the real monitor, in which Minecraft UI will be shown. (Please configure extra display settings, if you run on a virtual monitor.)

## 1. Setup NVIDIA GPU software (drivers and libraries) ##

> Note : When you just run the trained agent without training, GPU-utilized machine won't be needed and you can then skip settings in this section. (Please make sure to install ```tensorflow==2.4.1``` on CPUs, instead of ```tensorflow-gpu==2.4.1```.)

In this example, a large model (AttentionNet with ConvNet) will be used for training a RL agent.<br>
In order to speed up training, use GPU-utilized machine.

In this settings, we'll use CUDA version 11.0 and cuDNN versioin 8.0, because I will use TensorFlow 2.4.1.<br>
You should install the correct version of NVIDIA libraries, corresponding with TensorFlow version. (See [here](https://www.tensorflow.org/install/source#gpu) for details about compatible drivers in TensorFlow.)

First of all, install ```gcc``` and ```make``` tools for building utilities.

```
sudo apt-get update
sudo apt install -y gcc
sudo apt-get install -y make
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

## 2. Setup prerequisite software in Ubuntu ##

Make sure that Python 3 is installed on Ubuntu. (If not, please install Python 3 on Ubuntu.)

```
python3 -V
```

Install and upgrade pip3 as follows.

```
sudo apt-get update
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip
```

Install X remote desktop components, and start RDP service.<br>
After this settings, restart your computer.

```
sudo apt-get update
sudo apt-get install -y lxde
sudo apt-get install -y xrdp
/etc/init.d/xrdp start  # password is required
```

Allow (Open) inbound port 3389 (default RDP port) in network settings to enable your client to connect.

> Note : When you want to join into the same game with your own Minecraft client remotely, open Minecraft port 25565 too.

Install and setup Java (JDK) as follows. (JDK is needed to build Minecraft runtime.)

```
sudo apt-get install -y openjdk-8-jdk
echo -e "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> ~/.bashrc
source ~/.bashrc
```

## 3. Install and Setup Project Malmo ##

Install Project Malmo binaries, which has a modded Minecraft built by Microsoft Research.

```
# install prerequisite packages
pip3 install gym==0.21.0 lxml numpy pillow
# install malmo
pip3 install --index-url https://test.pypi.org/simple/ malmo==0.36.0
```

Expand Malmo bootstrap files as follows.<br>
All files will be deployed on ```./MalmoPython``` folder.

```
sudo apt-get install -y git
python3 -c "import malmo.minecraftbootstrap; malmo.minecraftbootstrap.download();"
```

Set ```MALMO_XSD_PATH``` environment variable as follows.

```
echo -e "export MALMO_XSD_PATH=$PWD/MalmoPlatform/Schemas" >> ~/.bashrc
source ~/.bashrc
```

Set the Malmo version in ```./MalmoPlatform/Minecraft/src/main/resources/version.properties``` file by running the following command. (If it's already set, there's nothing to do.)

```
cd MalmoPlatform/Minecraft
(echo -n "malmomod.version=" && cat ../VERSION) > ./src/main/resources/version.properties
cd ../..
```

## 4. Install Ray and RLlib framework ##

Install Ray framework with RLlib (which is used to run RL agent) with dependencies as follows.<br>
In this example, I have used TensorFlow for RLlib backend, but you can also use PyTorch backend.

```
pip3 install tensorflow-gpu==2.4.1 ray[default]==1.6.0 ray[rllib]==1.6.0 ray[tune]==1.6.0 attrs==19.1.0 pandas
```

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

## 6. Train an agent (Deep Reinforcement Learning) ##

Let's start training.

Login Ubuntu using remote desktop client and run the following commands **on monitor-attached shell** (such as, LXTerminal), because it will launch Minecraft UI.

Now, run the training script (```train.py```) as follows.<br>

```
cd minecraft-rl-pigchase-attention
python3 train.py --num_gpus 1
```

> Note : For the first time to run, all dependencies for building Minecraft (including Project Malmo's mod) are built and installed, and it will then take a while to start. Please be patient to wait.

> Note : For troubleshooting in building Minecraft or using monitor in Minecraft, see [here](https://github.com/tsmatz/minecraft-rl-example).

When you start the training code (```train.py```), you will see the agent's view in 160 x 120 Minecraft's screen. This frame pixels are then used by agent to train.

See "[Run Reinforcement Learning on Ray Cluster](https://tsmatz.wordpress.com/2021/10/08/rllib-reinforcement-learning-multiple-machines-ray-cluster/)", when you run on multiple workers in Ray cluster to speed up.

## 7. Run pre-trained agent

This repository also includes pre-trained checkpoint (```checkpoint/checkpoint-XXX```) in this repo.<br>
You can then check the result soon.

Run the following command to run the pre-trained agent, also **on monitor-attached shell** (such as, LXTerminal).

```
cd minecraft-rl-pigchase-attention
python3 run_agent.py
```

If you have your own trained checkpoint, you can also run your trained agent as follows.

```
python3 run_agent.py --checkpoint_file YOUR_OWN_CHECKPOINT_FILE_PATH
```
