import traci
import numpy as np
import random
import timeit
import os

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
    def __init__(self, Model, TrafficGen, CMD, maxSteps, greenDuration, yellowDuration, numStates, numActions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = CMD
        self._max_steps = maxSteps
        self._green_duration = greenDuration
        self._yellow_duration = yellowDuration
        self._num_states = numStates
        self._num_actions = numActions
        self._reward_episode = []
        self._queue_length_episode = []