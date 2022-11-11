from __future__ import absolute_import
from __future__ import print_function
import os
from shutil import copyfile
from Test_Simulation import Simulation
from Model import TestModel
from Helpers import Import_Test_Setup, Set_Test_Dir, Sumo_Settings, Save_and_Visualize


if __name__ == "__main__":

    config = Import_Test_Setup(configuration_file='Testing_Setup.ini')
    CMD = Sumo_Settings(config['gui'], config['sumocfgFileName'], config['maxSteps'])
    Path, plot_path = Set_Test_Dir(config['modelsPathName'], config['modelForTesting'])

    Model = TestModel(
        input_dim=config['numStates'],
        model_path=Path
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

   