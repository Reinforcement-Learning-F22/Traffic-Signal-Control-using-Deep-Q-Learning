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

def set_sumo(gui, sumocfgFileName, maxSteps):
    """
    Configure the parameters for SUMO
    """
    # It is necessery to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # settings for the visual mode    
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
 
    # to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join('intersection', sumocfgFileName), "--no-step-log", "true", "--waiting-time-memory", str(maxSteps)]

    return sumo_cmd