import itertools

import numpy as np
import pandas as pd

box_indices = np.array(list(itertools.product([0, 1, 2], [0, 1, 2])))


class Contradiction(Exception):
    pass


def load_grid():
    with open('extreme-2.txt') as fp:
        return np.array([[int(c.replace('.', '0')) for c in line.replace('\n', '')] for line in fp])


def discriminate_cell_by_cell(possibilities, grid):
    pivot = pd.DataFrame(grid).T.unstack()
    for number in range(1, 10):
        indices = zip(*pivot[pivot == number].index.labels)
        for row, col in indices:
            for i in range(9):
                possibilities[i, col] = possibilities[i, col] - {number}
                possibilities[row, i] = possibilities[row, i] - {number}
            box_corner = np.array([row, col]) // 3 * 3
            for index in map(tuple, box_indices + box_corner):
                possibilities[index] = possibilities[index] - {number}


def discriminate_in_group(possibilities, indices):
    indices = list(indices)
    group = (possibilities[index] for index in indices)
    values = list(itertools.chain(*group))
    counts = pd.Series(values).value_counts()
    results = set(counts[counts == 1].index)
    for index in indices:
        common = results & possibilities[index]
        if len(common) == 1:
            possibilities[index] = common
        elif len(common) > 1:
            raise Contradiction


def find_results(possibilities, grid):
    found = 0
    for index, possible_values in possibilities.items():
        if len(possible_values) == 1:
            found += 1
            grid[index] = next(iter(possible_values))
            possibilities[index] = set()
    return found


def solve_until_stuck(grid):
    possibilities = {}
    for index in itertools.product(range(9), range(9)):
        possibilities[index] = set() if grid[index] != 0 else set(range(1, 10))

    found = 1
    total_found = 0
    while found > 0:
        discriminate_cell_by_cell(possibilities, grid)
        for i, j in itertools.product(range(3), range(3)):
            box_corner = 3 * np.array([j, i])
            discriminate_in_group(possibilities, map(tuple, box_indices + box_corner))
        for j in range(9):
            discriminate_in_group(possibilities, [(j, i) for i in range(9)])
        for i in range(9):
            discriminate_in_group(possibilities, [(j, i) for j in range(9)])
        found = find_results(possibilities, grid)
        total_found += found
    return total_found, possibilities


def is_solved(grid):
    expected = set(range(1, 10))
    for i in range(9):
        if set(grid[:, i]) != expected:
            return False
        if set(grid[i, :]) != expected:
            return False
    for j, i in itertools.product(range(3), range(3)):
        box_grid = grid[3 * j:3 * (j + 1), 3 * i:3 * (i + 1)]
        if set(box_grid.flatten()) != expected:
            return False
    return True


def beam_search(grid, possibilities, past_tries, past_score):
    print('beam search level')
    top_possibilities = sorted(possibilities.items(), key=lambda item: len(item[1]))
    top_possibilities = list(item for item in top_possibilities if len(item[1]) > 0)
    results = []
    for index, values in top_possibilities:
        for value in values:
            sub_grid = grid.copy()
            sub_grid[index] = value
            try:
                found, sub_possibilities = solve_until_stuck(sub_grid)
            except Contradiction:
                continue
            tries = past_tries + [(index, value)]
            if is_solved(sub_grid):
                return [sub_grid, sub_possibilities, tries, 81]
            elif found > 0:
                print(tries, past_score + found)
                results.append([sub_grid, sub_possibilities, tries, past_score + found])

    # at this points beam search has reached a result so let's peek at n+1 step
    results = sorted(results, key=lambda result: result[3], reverse=True)
    for sub_grid, sub_possibilities, tries, score in results:
        sub_results = beam_search(sub_grid, sub_possibilities, tries, score)
        if len(sub_results) == 1 and sub_results[0][3] == 81:
            return sub_results[0]

    assert False, 'beam search failed to find solution'


grid = load_grid()
_, possibilities = solve_until_stuck(grid)
solution, *_ = beam_search(grid, possibilities, [], (grid != 0).sum())
print(solution)
