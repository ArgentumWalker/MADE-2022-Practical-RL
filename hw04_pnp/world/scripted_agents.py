import abc
import random
import numpy as np
import copy


class ScriptedAgent:
    @abc.abstractmethod
    def get_actions(self, state, team):
        pass

    @abc.abstractmethod
    def reset(self, initial_state, team):
        pass


class Dummy(ScriptedAgent):
    def __init__(self, num_predators=5):
        self.num_predators = num_predators
        self.random = random.Random()

    def get_actions(self, state, team):
        return [self.random.randint(0, 4) for i in range(self.num_predators)]

    def reset(self, initial_state, team):
        pass


class ClosestTargetAgent(ScriptedAgent):
    def __init__(self, num_predators=5):
        self.num_predators = num_predators
        self.distance_map = None
        self.action_map = None

    def get_actions(self, state, team):
        actions = [0 for _ in range(self.num_predators)]
        predators = {}
        preys = []
        preys_team = np.max(state[:, :, 0])
        if preys_team == team:
            preys_team = None
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                if state[x, y, 0] == team:
                    predators[state[x, y, 1]] = (x, y)
                    continue
                if (preys_team is None and state[x, y, 0] > 0) or (state[x, y, 0] == preys_team):
                    preys.append((x, y))
        if len(preys) > 0:
            for k, (xs, ys) in predators.items():
                target = preys[0]
                for p in preys:
                    if (self.distance_map[xs * state.shape[1] + ys, p[0] * state.shape[1] + p[1]] <
                            self.distance_map[xs * state.shape[1] + ys, target[0] * state.shape[1] + target[1]]):
                        target = p
                actions[k] = self.action_map[xs * state.shape[1] + ys, target[0] * state.shape[1] + target[1]]
        return actions

    def reset(self, initial_state, team):
        mask = np.zeros(initial_state.shape[:2], np.bool)
        mask[np.logical_or(np.logical_and(initial_state[:, :, 0] == -1, initial_state[:, :, 1] == 0),
                           initial_state[:, :, 0] >= 0)] = True
        mask = mask.reshape(-1)

        coords_amount = initial_state.shape[0] * initial_state.shape[1]
        self.distance_map = (coords_amount + 1) * np.ones((coords_amount, coords_amount))
        np.fill_diagonal(self.distance_map, 0.)
        self.distance_map[np.logical_not(mask)] = (coords_amount + 1)
        self.distance_map[:, np.logical_not(mask)] = (coords_amount + 1)

        indexes_helper = [
            [
                x * initial_state.shape[1] + (y + 1) % initial_state.shape[1],
                x * initial_state.shape[1] + (initial_state.shape[1] + y - 1) % initial_state.shape[1],
                ((initial_state.shape[0] + x - 1) % initial_state.shape[0]) * initial_state.shape[1] + y,
                ((x + 1) % initial_state.shape[0]) * initial_state.shape[1] + y
            ]
            for x in range(initial_state.shape[0]) for y in range(initial_state.shape[1])
        ]

        updated = True
        while updated:
            old_distances = copy.deepcopy(self.distance_map)
            for j in range(coords_amount):
                if mask[j]:
                    for i in indexes_helper[j]:
                        if mask[i]:
                            self.distance_map[j] = np.minimum(self.distance_map[j], self.distance_map[i] + 1)
            updated = (old_distances != self.distance_map).sum() > 0

        self.action_map = np.zeros((coords_amount, coords_amount), int)
        for j in range(coords_amount):
            self.action_map[j] = np.argmin(np.stack([self.distance_map[i] + 1 for i in indexes_helper[j]], axis=1),
                                           axis=1) + 1


class BrokenClosestTargetAgent(ClosestTargetAgent):
    def get_actions(self, state, team):
        actions = super().get_actions(state, team)
        for i in range(len(actions)):
            if random.random() < 0.10:
                actions[i] = 0
        return actions