import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        state = torch.tensor(np.array(state))
        return 0 # TODO

    def reset(self):
        pass

