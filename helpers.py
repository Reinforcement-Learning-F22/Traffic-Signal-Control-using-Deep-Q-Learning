import configparser
from sumolib import checkBinary
import numpy as np
import math
import random


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