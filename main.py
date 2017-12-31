import itertools
import typing as t

import numpy as np
import pandas as pd

box_indices = np.array(list(itertools.product([0, 1, 2], [0, 1, 2])))


class Contradiction(Exception):
    pass


class RuleBasedSolver:
    def __init__(self, grid: np.array):
        self.grid = grid
        self.possibilities: t.Dict[t.Tuple[int, int], t.Set[int]] = {}
        for index in itertools.product(range(9), range(9)):
            self.possibilities[index] = set() if grid[index] != 0 else set(range(1, 10))

    def discriminate_cell_by_cell(self):
        pivot: pd.Series = pd.DataFrame(self.grid).T.unstack()
        for number in range(1, 10):
            indices = zip(*pivot[pivot == number].index.labels)
            for row, col in indices:
                for i in range(9):
                    self.possibilities[i, col] = self.possibilities[i, col] - {number}
                    self.possibilities[row, i] = self.possibilities[row, i] - {number}
                box_corner = np.array([row, col]) // 3 * 3
                for index in map(tuple, box_indices + box_corner):
                    self.possibilities[index] = self.possibilities[index] - {number}

    def discriminate_in_group(self, indices: t.Iterable[t.Tuple[int, int]]):
        indices = list(indices)
        group = (self.possibilities[index] for index in indices)
        values = list(itertools.chain(*group))
        counts = pd.Series(values).value_counts()
        results = set(counts[counts == 1].index)
        for index in indices:
            common = results & self.possibilities[index]
            if len(common) == 1:
                self.possibilities[index] = common
            elif len(common) > 1:
                raise Contradiction

    def find_results(self):
        found = 0
        for index, possible_values in self.possibilities.items():
            if len(possible_values) == 1:
                found += 1
                self.grid[index] = next(iter(possible_values))
                self.possibilities[index] = set()
        return found

    def solve_until_stuck(self):
        found = 1
        total_found = 0
        while found > 0:
            self.discriminate_cell_by_cell()
            for i, j in itertools.product(range(3), range(3)):
                box_corner = 3 * np.array([j, i])
                self.discriminate_in_group(map(tuple, box_indices + box_corner))
            for j in range(9):
                self.discriminate_in_group((j, i) for i in range(9))
            for i in range(9):
                self.discriminate_in_group((j, i) for j in range(9))
            found = self.find_results()
            total_found += found
        return total_found

    def is_solved(self):
        expected = set(range(1, 10))
        for i in range(9):
            if set(self.grid[:, i]) != expected:
                return False
            if set(self.grid[i, :]) != expected:
                return False
        for j, i in itertools.product(range(3), range(3)):
            box_grid = self.grid[3 * j:3 * (j + 1), 3 * i:3 * (i + 1)]
            if set(box_grid.flatten()) != expected:
                return False
        return True


class BeamSearchStep:
    def __init__(self,
                 grid: np.array,
                 possibilities: t.Dict[t.Tuple[int, int], t.Set[int]],
                 path: t.List[t.Tuple[t.Tuple[int, int], int]],
                 score: int,
                 ):
        self.grid = grid
        self.possibilities = possibilities
        self.path = path
        self.score = score


class BeamSearchSolver:
    def __init__(self, beam_size=5):
        self.beam_size = beam_size

    def _walk(self, step: BeamSearchStep) -> t.List[BeamSearchStep]:
        print('beam search step')
        top_possibilities = [(index, values) for index, values in step.possibilities.items() if len(values) > 0]
        top_possibilities = sorted(top_possibilities, key=lambda item: len(item[1]))
        top_possibilities = top_possibilities[:self.beam_size]
        steps: t.List[BeamSearchStep] = []
        for index, values in top_possibilities:
            for value in values:
                sub_grid = step.grid.copy()
                sub_grid[index] = value
                path = step.path + [(index, value)]
                solver = RuleBasedSolver(sub_grid)
                try:
                    found = solver.solve_until_stuck()
                except Contradiction:
                    print(path, -1)
                    continue
                sub_possibilities = solver.possibilities
                if solver.is_solved():
                    print(path, 81)
                    return [BeamSearchStep(sub_grid, sub_possibilities, path, 81)]
                elif found > 0:
                    print(path, step.score + found)
                    steps.append(BeamSearchStep(sub_grid, sub_possibilities, path, step.score + found))

        # at this points beam search has reached a result so let's peek at n+1 step
        steps = sorted(steps, key=lambda step: step.score, reverse=True)
        for i, step in enumerate(steps):
            sub_steps = self._walk(step)
            if len(sub_steps) == 1 and sub_steps[0].score == 81:
                return sub_steps

        return steps

    def solve(self, grid):
        grid = grid.copy()
        solver = RuleBasedSolver(grid)
        solver.solve_until_stuck()
        step = BeamSearchStep(grid, solver.possibilities, [], (grid != 0).sum())
        results = self._walk(step)
        try:
            return results[0].grid
        except IndexError:
            return None


def load_grid():
    with open('extreme-3.txt') as fp:
        return np.array([[int(c.replace('.', '0')) for c in line.replace('\n', '')] for line in fp])


def main():
    grid = load_grid()
    solution = BeamSearchSolver(beam_size=3).solve(grid)
    print(solution)


if __name__ == '__main__':
    main()
