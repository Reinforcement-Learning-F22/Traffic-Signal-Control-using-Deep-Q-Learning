from __future__ import absolute_import
from __future__ import print_function
import os
import datetime
from shutil import copyfile
from Train_Simulation import Simulation
from Model import TrainModel
from Helpers import Import_Train_Setup, Set_Train_Dir, Sumo_Settings, Save_and_Visualize


if __name__ == "__main__":

    config = Import_Train_Setup(configuration_file='Training_Setup.ini')
    CMD = Sumo_Settings(config['gui'], config['sumocfgFileName'], config['maxSteps'])
    Path = Set_Train_Dir(config['modelsPathName'])

    Model = TrainModel(
        config['numLayers'], 
        config['layerWidth'], 
        config['batchSize'], 
        config['learningRate'], 
        config['numStates'], #input_dim
        config['numActions'], #output_dim
        config['maxMemorySize'], 
        config['minMemorySize']
    )
    Simulation = Simulation(
        Model,
        CMD,
        config['gamma'],
        config['maxSteps'],
        config['numCars'],
        config['greenDuration'],
        config['yellowDuration'],
        config['numStates'],
        config['numActions'],
        config['trainingEpochs']
    )
    