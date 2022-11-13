from .base import StochasticMapLoader
import numpy as np
import abc
from queue import PriorityQueue


class TwoTeamMapLoader(StochasticMapLoader):
    def __init__(self, size=40, spawn_radius=8, preys_num=100, spawn_points=10, spawn_attempts=30):
        self.size = size
        self.spawn_radius = spawn_radius
        self.spawn_points = spawn_points
        self.preys_num = preys_num // 2
        self.spawn_attempts = spawn_attempts

    def _generate(self):
        generated = False
        map = np.zeros((self.size, self.size), int)
        while not generated:
            map = np.zeros((self.size, self.size), int)
            self._generate_rocks(map)
            generated = self._generate_entities(map)
        mask = np.tri(self.size)
        other_part = (1 - mask) * map.T
        other_part[other_part == 1] = 2
        map = (mask * map + other_part).astype(np.int)
        return map

    @abc.abstractmethod
    def _generate_rocks(self, map):
        pass

    def _generate_entities(self, map):
        spawn_radius = self.size // 5
        spawn_shift_top = 3 * self.size // 5
        spawn_shift_left = spawn_radius
        max_spawn_points = (map[spawn_shift_top:spawn_shift_top+spawn_radius-1,
                            spawn_shift_left:spawn_shift_left+spawn_radius-1] == 0).sum()
        for _ in range(max_spawn_points):
            attempt = 0
            x, y = self.random.randint(0, spawn_radius), self.random.randint(0, spawn_radius)
            while map[spawn_shift_top+x][spawn_shift_left+y] != 0:
                x, y = self.random.randint(0, spawn_radius), self.random.randint(0, spawn_radius)
                attempt += 1
                if attempt > self.spawn_attempts:
                    return False
            map[spawn_shift_top + x, spawn_shift_left + y] = 1
        max_preys = min(self.preys_num, (map == 0).sum())

        for _ in range(max_preys):
            x = self.random.randint(0, self.size)
            y = self.random.randint(0, self.size)
            if x < y:
                x, y = y, x
            attempt = 0
            while map[x][y] != 0 or x == y:
                x = self.random.randint(0, self.size)
                y = self.random.randint(0, self.size)
                if x < y:
                    x, y = y, x
                attempt += 1
                if attempt > self.spawn_attempts:
                    return False
            map[x, y] = -2
        return True


class TwoTeamRocksMapLoader(TwoTeamMapLoader):
    def __init__(self, size=40, spawn_radius=8, preys_num=100, spawn_points=10, rock_spawn_proba=0.15,
                 additional_rock_spawn_proba=0.2, spawn_attempts=30):
        super().__init__(size, spawn_radius, preys_num, spawn_points, spawn_attempts)
        self.rock_spawn_proba = rock_spawn_proba
        self.additional_rock_spawn_proba = additional_rock_spawn_proba

    def _generate_rocks(self, map):
        prev_rocks = np.zeros((self.size, self.size))
        rocks = self.random.rand(self.size, self.size) < self.rock_spawn_proba
        new_rocks = rocks
        while (prev_rocks != rocks).sum() > 0:
            candidates = np.logical_or(np.logical_or(np.roll(new_rocks, 1, 0), np.roll(new_rocks, -1, 0)),
                                       np.logical_or(np.roll(new_rocks, 1, 1), np.roll(new_rocks, -1, 1)))
            new_rocks = np.logical_and(candidates,
                                       self.random.rand(self.size, self.size) < self.additional_rock_spawn_proba)
            prev_rocks = rocks
            rocks = np.logical_or(rocks, new_rocks)

        map[rocks] = -1


class TwoTeamLabyrinthMapLoader(TwoTeamMapLoader):
    def __init__(self, size=40, spawn_radius=8, preys_num=100, spawn_points=10, additional_links_max=12,
                 additional_links_min=0, spawn_attempts=30):
        super().__init__(size, spawn_radius, preys_num, spawn_points, spawn_attempts)
        self.additional_links_max = additional_links_max
        self.additional_links_min = additional_links_min

    def _generate_rocks(self, map):
        cells = np.zeros((self.size // 4, self.size // 4, 4))
        expandable_cells = PriorityQueue()
        expandable_cells.put((self.random.rand(), (self.random.randint(0, self.size // 4),
                                                  self.random.randint(0, self.size // 4))))
        direction_helper = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])

        while expandable_cells.qsize() > 0:
            _, (x, y) = expandable_cells.get()
            directions = (self.size + direction_helper + np.array([[x, y] for _ in range(4)])) % (self.size // 4)
            available_directions = []
            for i, (tx, ty) in enumerate(directions):
                if cells[tx, ty].sum() == 0 and not (x == y and i % 2 == 1):
                    available_directions.append(i)
            if len(available_directions) == 0:
                continue
            i = available_directions[self.random.randint(0, len(available_directions))]
            tx, ty = directions[i]
            cells[x, y, i] = 1
            cells[tx, ty, 3 - i] = 1
            cells[y, x, (5 - i) % 4] = 1
            cells[ty, tx, (i + 2) % 4] = 1
            if tx != ty:
                expandable_cells.put((self.random.rand(), (tx, ty)))
            if len(available_directions) > 1 and x != y:
                expandable_cells.put((self.random.rand(), (x, y)))

        for _ in range(self.random.randint(self.additional_links_min, self.additional_links_max + 1)):
            x, y = self.random.randint(0, self.size // 4), self.random.randint(0, self.size // 4)
            i = self.random.randint(0, 4)
            if x < y:
                x, y = y, x
            while cells[x, y, i] > 0 or x == y:
                x, y = self.random.randint(0, self.size // 4), self.random.randint(0, self.size // 4),
                if x < y:
                    x, y = y, x
                i = self.random.randint(0, 4)
            cells[x, y, i] = 1
            tx, ty = (self.size + direction_helper[i] + np.array([x, y])) % (self.size // 4)
            cells[tx, ty, 3 - i] = 1

        for x in range(self.size // 4):
            if cells[x, -1, 1] == 1:
                cells[0, x, 3] = 1
            if cells[0, x, 3] == 1:
                cells[x, -1, 1] = 1

        for x in range(self.size // 4):
            for y in range(self.size // 4):
                map[4 * x:4 * (x + 1), 4 * y:4 * (y + 1)] = -1
                map[4 * x + 1:4 * (x + 1) - 1, 4 * y + 1:4 * (y + 1) - 1] = 0
                if cells[x, y, 3] > 0:
                    map[4 * x, 4 * y + 1:4 * (y + 1) - 1] = 0
                if cells[x, y, 0] > 0:
                    map[4 * (x + 1) - 1, 4 * y + 1:4 * (y + 1) - 1] = 0
                if cells[x, y, 2] > 0:
                    map[4 * x + 1:4 * (x + 1) - 1, 4 * y] = 0
                if cells[x, y, 1] > 0:
                    map[4 * x + 1:4 * (x + 1) - 1, 4 * (y + 1) - 1] = 0