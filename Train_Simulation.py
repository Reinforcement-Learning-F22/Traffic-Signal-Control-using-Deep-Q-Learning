import traci
import numpy as np
import random
import timeit
from helpers import Traffic_Route_Generator

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, CMD, gamma, maxSteps, numCars, greenDuration, yellowDuration, numStates, numActions, trainingEpochs):
        self._Model = Model
        self._sumo_cmd = CMD
        self._gamma = gamma
        self._maxSteps = maxSteps
        self._numCars = numCars
        self._greenDuration = greenDuration
        self._yellowDuration = yellowDuration
        self._numStates = numStates
        self._numActions = numActions
        self._step = 0
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = trainingEpochs

