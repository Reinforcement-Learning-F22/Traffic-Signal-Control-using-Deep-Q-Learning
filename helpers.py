import configparser
from sumolib import checkBinary
import os
import sys
import numpy as np
import math


def Import_Train_Setup(configuration_file):
    """
    Import training parameters from train setup configuration file. 
    """
    content = configparser.ConfigParser()
    content.read(configuration_file)
    config = {}
    
    # Simulation Parameters
    config['gui'] = content['simulation'].getboolean('gui')
    config['numEpisodes'] = content['simulation'].getint('numEpisodes')
    config['maxSteps'] = content['simulation'].getint('maxSteps')
    config['numCars'] = content['simulation'].getint('numCars')
    config['greenDuration'] = content['simulation'].getint('greenDuration')
    config['yellowDuration'] = content['simulation'].getint('yellowDuration')
    
    # Model Parameters
    config['numLayers'] = content['model'].getint('numLayers')
    config['layerWidth'] = content['model'].getint('layerWidth')
    config['batchSize'] = content['model'].getint('batchSize')
    config['learningRate'] = content['model'].getfloat('learningRate')
    config['trainingEpochs'] = content['model'].getint('trainingEpochs')
    
    # Memory Parameters 
    config['minMemorySize'] = content['memory'].getint('minMemorySize')
    config['maxMemorySize'] = content['memory'].getint('maxMemorySize')

    # Agent Parameters 
    config['numStates'] = content['agent'].getint('numStates')
    config['numActions'] = content['agent'].getint('numActions')
    config['gamma'] = content['agent'].getfloat('gamma')

    # Directories  
    config['modelsPathName'] = content['dir']['modelsPathName']
    config['sumocfgFileName'] = content['dir']['sumocfgFileName']
    return config


def Import_Test_Setup(configuration_file):

    """
    Import testing parameters from test setup configuration file. 
    """
    content = configparser.ConfigParser()
    content.read(configuration_file)
    config = {}

    # Simulation Parameters
    config['gui'] = content['simulation'].getboolean('gui')
    config['maxSteps'] = content['simulation'].getint('maxSteps')
    config['numCars'] = content['simulation'].getint('numCars')
    config['episodeSeed'] = content['simulation'].getint('episodeSeed')
    config['yellowDuration'] = content['simulation'].getint('yellowDuration')
    config['greenDuration'] = content['simulation'].getint('greenDuration')
    
    # Agent Parameters 
    config['numStates'] = content['agent'].getint('numStates')
    config['numActions'] = content['agent'].getint('numActions')
    
    # Directories      
    config['modelsPathName'] = content['dir']['modelsPathName']
    config['sumocfgFileName'] = content['dir']['sumocfgFileName']
    config['modelForTesting'] = content['dir'].getint('modelForTesting') 
    return config


def Set_Train_Dir(modelsPathName):
    """
    Create a path for the new model path and increment the path name based on the previously model paths created. 
    """
    folderPath = os.path.join(os.getcwd(), modelsPathName, '')
    os.makedirs(os.path.dirname(folderPath), exist_ok=True)

    dir_content = os.listdir(folderPath)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    dataPath = os.path.join(folderPath, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(dataPath), exist_ok=True)
    return dataPath 


def Set_Test_Dir(modelsPathName, model_N):
    """
    Returns the path of the model in which the model number has been provided as an argument.
    """
    folderPath = os.path.join(os.getcwd(), modelsPathName, 'model_'+str(model_N), '')

    if os.path.isdir(folderPath):    
        plotPath = os.path.join(folderPath, 'test', '')
        os.makedirs(os.path.dirname(plotPath), exist_ok=True)
        return folderPath, plotPath
    else: 
        sys.exit('The model number specified does not exist in the models folder')


def Traffic_Route_Generator(numCars, maxSteps, seed):
    """
    Generation of the route of every car for one episode
    """
    np.random.seed(seed)  # make tests reproducible

    # Car generation with weibull distribution
    timings = np.random.weibull(2, numCars)
    timings = np.sort(timings)

    # Distribution reshape to fit the interval 0:max_steps
    car_gen_steps = []
    old_min = math.floor(timings[1])
    old_max = math.ceil(timings[-1])
    new_min = 0
    new_max = maxSteps
    for value in timings:
        car_gen_steps = np.append(car_gen_steps, ((new_max - new_min) / (old_max - old_min)) * (value - old_max) + new_max)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

    # produce the file for cars generation, one car per line
    with open("Sumo_environment/episode_routes.rou.xml", "w") as routes:
        print("""<routes>
        <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

        <route id="W_N" edges="W2TL TL2N"/>
        <route id="W_E" edges="W2TL TL2E"/>
        <route id="W_S" edges="W2TL TL2S"/>
        <route id="N_W" edges="N2TL TL2W"/>
        <route id="N_E" edges="N2TL TL2E"/>
        <route id="N_S" edges="N2TL TL2S"/>
        <route id="E_W" edges="E2TL TL2W"/>
        <route id="E_N" edges="E2TL TL2N"/>
        <route id="E_S" edges="E2TL TL2S"/>
        <route id="S_W" edges="S2TL TL2W"/>
        <route id="S_N" edges="S2TL TL2N"/>
        <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

    for car_counter, step in enumerate(car_gen_steps):
        straight_or_turn = np.random.uniform()
        if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
            route_straight = np.random.randint(1, 5)  # choose a random source & destination
            if route_straight == 1:
                print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            elif route_straight == 2:
                print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            elif route_straight == 3:
                print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            else:
                print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
        else:  # car that turn -25% of the time the car turns
            route_turn = np.random.randint(1, 9)  # choose random source source & destination
            if route_turn == 1:
                print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            elif route_turn == 2:
                print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            elif route_turn == 3:
                print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            elif route_turn == 4:
                print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            elif route_turn == 5:
                print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            elif route_turn == 6:
                print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            elif route_turn == 7:
                print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            elif route_turn == 8:
                print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

    print("</routes>", file=routes)
