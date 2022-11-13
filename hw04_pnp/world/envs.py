import copy

import numpy as np


class OnePlayerEnv:
    def __init__(self, realm):
        self.realm = realm

    def step(self, actions):
        self.realm.set_actions(actions, 0)
        state = copy.deepcopy(self.realm.world.map)
        info = {
            "eaten": copy.deepcopy(self.realm.eaten),
            "preys": copy.deepcopy([prey.get_state() for prey in self.realm.world.preys]),
            "predators": copy.deepcopy([predator.get_state() for predator in self.realm.world.teams[0].values()]),
            "scores": copy.deepcopy(self.realm.team_scores)
        }
        done = self.realm.done or len(self.realm.world.eaten_preys) == len(self.realm.world.preys)
        return state, done, info

    def reset(self, seed=None):
        self.realm.reset(seed)
        state = copy.deepcopy(self.realm.world.map)
        info = {
            "eaten": copy.deepcopy(self.realm.eaten),
            "preys": copy.deepcopy([prey.get_state() for prey in self.realm.world.preys]),
            "predators": copy.deepcopy([predator.get_state() for predator in self.realm.world.teams[0].values()]),
            "scores": copy.deepcopy(self.realm.team_scores)
        }
        return state, info


class VersusBotEnv(OnePlayerEnv):
    def step(self, actions):
        state, done, info = super().step(actions)
        info["enemy"] = copy.deepcopy([predator.get_state() for predator in self.realm.world.teams[1].values()])
        #done = self.realm.step_num >= self.realm.step_limit
        return state, done, info

    def reset(self, seed=None):
        state, info = super().reset(seed)
        info["enemy"] = copy.deepcopy([predator.get_state() for predator in self.realm.world.teams[1].values()])
        return state, info


class TwoPlayerEnv:
    def __init__(self, realm):
        self.realm = realm

    def step(self, actions1, actions2):
        self.realm.set_actions(actions1, 0)
        actions2 = [
            (5 - a if 0 < a <= 4 else a)
            for a in actions2
        ]
        self.realm.set_actions(actions2, 1)
        done = self.realm.done or len(self.realm.world.eaten_preys) == len(self.realm.world.preys)
        state1, info1, state2, info2 = self._compute_states_and_infos()
        return (state1, done, info1), (state2, done, info2)

    def reset(self, seed=None):
        self.realm.reset(seed)
        state1, info1, state2, info2 = self._compute_states_and_infos()
        return (state1, info1), (state2, info2)

    def _compute_states_and_infos(self):

        state1 = copy.deepcopy(self.realm.world.map)
        info1 = {
            "eaten": copy.deepcopy(self.realm.eaten),
            "preys": copy.deepcopy([prey.get_state() for prey in self.realm.world.preys]),
            "predators": copy.deepcopy([predator.get_state() for predator in self.realm.world.teams[0].values()]),
            "enemy": copy.deepcopy([predator.get_state() for predator in self.realm.world.teams[1].values()]),
            "scores": copy.deepcopy(self.realm.team_scores)
        }
        state2 = copy.deepcopy(np.transpose(state1, (1, 0, 2)))
        mask = copy.deepcopy(state2[:, :, 0])
        state2[:, :, 0][mask == 0] = 1
        state2[:, :, 0][mask == 1] = 0
        info2 = {
            "eaten": dict([(((2 if t1 == 2 else 1-t1), i1), ((2 if t2 == 2 else 1-t2), i2))
                            for (t1, i1), (t2, i2) in self.realm.eaten.items()]),
            "preys": copy.deepcopy([prey.get_state() for prey in self.realm.world.preys]),
            "predators": copy.deepcopy([predator.get_state() for predator in self.realm.world.teams[1].values()]),
            "enemy": copy.deepcopy([predator.get_state() for predator in self.realm.world.teams[0].values()]),
            "scores": copy.deepcopy(list(reversed(self.realm.team_scores)))
        }
        for i in range(len(info2["preys"])):
            info2["preys"][i]["x"], info2["preys"][i]["y"] = info2["preys"][i]["y"], info2["preys"][i]["x"]
        for i in range(len(info2["predators"])):
            info2["predators"][i]["team"] = 0
            info2["predators"][i]["x"], info2["predators"][i]["y"] = info2["predators"][i]["y"], info2["predators"][i]["x"]
        for i in range(len(info2["enemy"])):
            info2["enemy"][i]["team"] = 1
            info2["enemy"][i]["x"], info2["enemy"][i]["y"] = info2["enemy"][i]["y"], info2["enemy"][i]["x"]
        return state1, info1, state2, info2