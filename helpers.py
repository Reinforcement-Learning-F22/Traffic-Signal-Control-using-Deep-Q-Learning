import configparser
from sumolib import checkBinary
import os
import sys


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

def Test_Dir(modelsPathName, model_N):
    """
    Returns the path of the model in which the model number has been provided as an argument 
    """
    folderPath = os.path.join(os.getcwd(), modelsPathName, 'model_'+str(model_N), '')

    if os.path.isdir(folderPath):    
        plot_path = os.path.join(folderPath, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return folderPath, plot_path
    else: 
        sys.exit('The model number specified does not exist in the models folder')
