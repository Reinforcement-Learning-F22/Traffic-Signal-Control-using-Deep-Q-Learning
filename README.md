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
    ```
    sudo ubuntu-drivers autoinstall
    ```
2. Restart your machine and check if graphic card is installed.
    ```
    sudo reboot
    nvidia-smi
    ```
    Please Note the CUDA Version on Top Right. This is required to follow correct CUDA version.

3. Download and Install CUDA Toolkit from [here](https://developer.nvidia.com/cuda-toolkit-archive).
4. Download the correspomding cuDNN for installedCUDA by signing up on [Nvidia](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse804-110).
5. Install cuDNN by extracting the contents of cuDNN into the Toolkit path installed in Step 3. There will be files that you have to replace in CUDA Toolkit Directory.
6. Check the path variables if CUDA_HOME is present and the toolkit paths are available.
7. Install Tensorflow. Tensorflow by default comes with GPU support,so no need to install tensorflow-gpu specifically. Run below in Terminal:
    ```
    sudo apt update
    sudo apt install python3-pip
    pip install tensorflow
    ```
8. Check Tensorflow and Check GPU is detected by Tensorflow. Run below in Terminal:
    Go to python console using ```python3``` and type the following.
    ```
    import tensorflow as tf
    tf.config.list_physical_devices('GPU')
    ```
    You should be able to see "[PhysicalDevice(name=’ physical_device:GPU:0', device_type='GPU')]".
    
I went through this approach for days and it turned out to be very hard because of versions compatibility of CUDA, cuDNN, and Tensorflow-gpu. It is really a headache and there is a probability of 1% that this process will go right for you!
So, better to dump pip and use conda instead.

**Conda Approach:**

1. First remove the pre-installed CUDA and Nvidia drivers to avoid CUDA and cuDNN versions incompatibility by running the follwoing: 
    ```
    sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" "*nvidia*"
    sudo apt-get purge nvidia*
    sudo apt-get autoremove
    sudo apt-get autoclean
    sudo rm -rf /usr/local/cuda*
    ```
2. Then reinstall he recommended Nvidia drivers for your machine.
    ```
    sudo ubuntu-drivers autoinstall
    ```
3. Download ([Anaconda](https://www.anaconda.com/distribution/#download-section)) and install it by running the following command in the terminal of download directory. 
    ```
    bash Anaconda3-2022.10-Linux-x86_64.sh
    ```
4. Create an environment first named with ‘tf_gpu’ and install all the packages required by tensorflow-gpu including the cuda and cuDNN compatible verisons.
    ```
    conda create --name tf_gpu
    activate tf_gpu
    conda install tensorflow-gpu
    ```
5. Testing your Tensorflow installation. Open the terminal and activiate the environment as follows.
    ```
    conda activiate tf_gpu
    ```
    Go to python console using ```python``` and type the following.
    ```
    import tensorflow as tf
    tf.config.list_physical_devices('GPU')
    ```
    ![image](https://user-images.githubusercontent.com/90580636/200055403-ad36db40-f9be-4cdd-8fdd-0ea0afa1535f.png)

Now everything should work properly ✔️.

**Libraries Installation:**
Create sumolib virtual environment:
```
conda create -n sumolib
conda activate sumolib
```
Install pip to sumo venv directory:
```
conda install pip
```
Find your anaconda directory, and find the actual sumolib env. It should be somewhere like /anaconda/envs/sumolib/ and 
install new packages:
```
pip install sumolib
```
Activate Tensorflow_gpu and install the following libraries:
```
conda activiate tf_gpu
pip install sumolib
pip install traci
pip install pydot
sudo apt install graphviz
```

## Code Structure

**Train_Main.py** is the main file in our repo.

The main file is training_main.py. On each iteration, it handles the main loop that starts an episode. It also saves the network weights as well as three plots: the negative reward plot, the cumulative wait time plot, and the average queues plot.

The algorithm is divided into classes that handle various aspects of training.

- Two different model classes are defined in the **[model.py](https://github.com/Reinforcement-Learning-F22/TrafficSignalControl/blob/main/Model.py)** file: one used only during training (**TrainModel**) and one used only during testing (**TestModel**). Each Model class defines everything about the deep neural network and includes some functions for training and predicting outputs.

- The **Simulation** class is in charge of the simulation. The function *run*, in particular, enables the simulation of a single episode. Other functions are also used during run to interact with **SUMO**, such as retrieving the environment's state (*get_state*), setting the next green light phase (*set_green_phase*), or preprocessing the data to train the neural network (*replay*). **[Train_Simulation.py](https://github.com/Reinforcement-Learning-F22/TrafficSignalControl/blob/main/Train_Simulation.py)** and **[Test_Simulation.py](https://github.com/Reinforcement-Learning-F22/TrafficSignalControl/blob/main/Test_Simulation.py)** each contain a slightly different **Simulation** class. Which one is loaded depends on whether we are in the training or testing phase.


The **[Sumo_environment](https://github.com/Reinforcement-Learning-F22/TrafficSignalControl/tree/main/Sumo_environment)** folder contains a file called **environment.net.xml(https://github.com/Reinforcement-Learning-F22/TrafficSignalControl/blob/main/Sumo_environment/environment.net.xml)**, which defines the structure of the environment and was created with SUMO NetEdit. The other file, sumo config.sumocfg, is a linker between the environment and route files.


## Training and Testing Settings

## Model Training and Testing

## Deep Q-Learning Agent 

## Acknowledgement
[Deep Q-Learning Agent for Traffic Signal Control](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control)

## References
[Tensorflow GPU Installation](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc)


    
    
    
