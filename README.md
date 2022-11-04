# Adaptive Traffic Signal Control using Deep Q-Learning Learning
In this project, we aim to implementing a learning algorithm that will allow traffic control devices to study traffic patterns/behaviors for a given intersection and optimize traffic flow by altering stoplight timing. With the help of Q-Learning technique, where an agent, based on the given state, selects an appropriate action for the intersection in order to maximize present and future rewards. 

## Environment Setup 
Please note that NVIDIA GPU is strongly recommended to run the algorithm. Below are the easiest steps in order to run the code from scratch in your machine.

### 1. Download Simulation of Urban MObility ([SUMO](https://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/)). 
In short, open your terminal, and type the following commands:
    
    sudo add-apt-repository ppa:sumo/stable
    sudo apt-get update
    sudo apt --fix-broken install
    sudo apt-get install sumo sumo-tools sumo-doc
    
### 2. Tensorflow GPU installation:
**Conventional Approach:**

    1. Install the recommended Nvidia-drivers for your system:
    
        sudo ubuntu-drivers autoinstall
    
    2. Restart your machine and check if graphic card is installed.
    
        sudo reboot
        nvidia-smi
    
        Please Note the CUDA Version on Top Right. This is required to follow correct CUDA version.
        
    3. Download and Install CUDA Toolkit from [here](https://developer.nvidia.com/cuda-toolkit-archive).
    4. Download the correspomding cuDNN for installedCUDA by signing up on [Nvidia](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse804-110).
    5. Install cuDNN by extracting the contents of cuDNN into the Toolkit path installed in Step 3. There will be files that you have to replace in CUDA Toolkit Directory.
    6. Check the path variables if CUDA_HOME is present and the toolkit paths are available.
    7. Install Tensorflow. Tensorflow by default comes with GPU support,so no need to install tensorflow-gpu specifically. Run below in Terminal:
    
        sudo apt update
        sudo apt install python3-pip
        pip install tensorflow
        
    8. Check Tensorflow and Check GPU is detected by Tensorflow. Run below in Terminal:
    &nbsp;&nbsp;&nbsp;&nbsp;Go to python console using```python3``` and type the following.
    &nbsp;&nbsp;&nbsp;&nbsp;```import tensorflow as tf```
    &nbsp;&nbsp;&nbsp;&nbsp;```tf.config.list_physical_devices('GPU')```
    &nbsp;&nbsp;&nbsp;&nbsp;You should be able to see "[PhysicalDevice(name=’ physical_device:GPU:0', device_type='GPU')]".
    
I went through this approach for days and it turned out to be very hard because of versions compatibility of CUDA, cuDNN, and Tensorflow-gpu. It is really a headache and there is a probability of 1% that this process will go right for you!
So, better to dump pip and use conda instead.

**Using Conda:**
1- First remove the pre-installed CUDA and Nvidia drivers to avoid CUDA and cuDNN versions incompatibility by running the follwoing: 
    &nbsp;&nbsp;&nbsp;&nbsp;```sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" "*nvidia*"```
    &nbsp;&nbsp;&nbsp;&nbsp;```sudo apt-get purge nvidia*```
    &nbsp;&nbsp;&nbsp;&nbsp;```sudo apt-get autoremove```
    &nbsp;&nbsp;&nbsp;&nbsp;```sudo apt-get autoclean```
    &nbsp;&nbsp;&nbsp;&nbsp;```sudo rm -rf /usr/local/cuda*```
2- Then reinstall he recommended Nvidia drivers for your machine.
    &nbsp;&nbsp;&nbsp;&nbsp;```sudo ubuntu-drivers autoinstall``` 
3- Download ([Anaconda](https://www.anaconda.com/distribution/#download-section)) and install it by running the following command in the terminal of download directory. 
    &nbsp;&nbsp;&nbsp;&nbsp;```bash Anaconda3-2022.10-Linux-x86_64.sh```
4- Create an environment first named with ‘tf_gpu’ and install all the packages required by tensorflow-gpu including the cuda and cuDNN compatible verisons.
&nbsp;&nbsp;&nbsp;&nbsp;```conda create --name tf_gpu```
&nbsp;&nbsp;&nbsp;&nbsp;```activate tf_gpu```
&nbsp;&nbsp;&nbsp;&nbsp;```conda install tensorflow-gpu```
5- Testing your Tensorflow installation. Open the terminal and activiate the environment as follows.
    &nbsp;&nbsp;&nbsp;&nbsp;```conda activiate tf_gpu```
    &nbsp;&nbsp;&nbsp;&nbsp;Go to python console using```python``` and type the following.
    &nbsp;&nbsp;&nbsp;&nbsp;```import tensorflow as tf```
    &nbsp;&nbsp;&nbsp;&nbsp;```tf.config.list_physical_devices('GPU')```

Now everything should work properly ✅.

**Libraries Installation:**
Create sumolib virtual environment:
    &nbsp;&nbsp;&nbsp;&nbsp;```conda create -n sumolib```
    &nbsp;&nbsp;&nbsp;&nbsp;```conda activate sumolib```
Install pip to sumo venv directory:
    &nbsp;&nbsp;&nbsp;&nbsp;```conda install pip```
Find your anaconda directory, and find the actual sumolib env. It should be somewhere like /anaconda/envs/sumolib/ and 
install new packages:
    &nbsp;&nbsp;&nbsp;&nbsp;```pip install sumolib```
Activate Tensorflow_gpu and install the following libraries:
    &nbsp;&nbsp;&nbsp;&nbsp;```conda activiate tf_gpu```
    &nbsp;&nbsp;&nbsp;&nbsp;```pip install sumolib```
    &nbsp;&nbsp;&nbsp;&nbsp;```pip install traci```
    &nbsp;&nbsp;&nbsp;&nbsp;```pip install pydot```
    &nbsp;&nbsp;&nbsp;&nbsp;```sudo apt install graphviz```

## Code Structure

## Training and Testing Settings

## Model Training and Testing

## Deep Q-Learning Agent 

## Acknowledgement
[Deep Q-Learning Agent for Traffic Signal Control](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control)

## References
[Tensorflow GPU Installation](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc)


    
    
    
