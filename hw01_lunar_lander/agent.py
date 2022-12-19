import numpy as np
import torch

from .train import DeepQNetworkModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    def __init__(self):
        self.model = DeepQNetworkModel(8, 4, 64)
        weights = torch.load(__file__[:-8] + "/agent.pth")
        self.model.load_state_dict(weights)
        self.model.to(DEVICE)
        self.model.eval()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = np.argmax(self.model(state).cpu().numpy())
        return action
