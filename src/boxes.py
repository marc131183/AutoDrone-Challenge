import numpy as np

from PIL import Image
from typing import List, Tuple


class Island:
    def __init__(self, i: int, j: int) -> None:
        self.min_row: int = i
        self.max_row: int = i
        self.min_col: int = j
        self.max_col: int = j
        self.size: int = 1

    def extendIsland(self, i: int, j: int) -> None:
        self.size += 1
        if i < self.min_row:
            self.min_row = i
        elif i > self.max_row:
            self.max_row = i
        if j < self.min_col:
            self.min_col = j
        elif j > self.max_col:
            self.max_col = j

    def __str__(self) -> str:
        return "Island object - size: {}, box: ({}, {}, {}, {})".format(
            self.size, self.min_row, self.min_col, self.max_row, self.max_col
        )


def getNeighbours(
    map_plan: np.ndarray, shape: Tuple[int, int], i: int, j: int
) -> List[Tuple[int, int]]:
    neighbours: List[Tuple[int, int]] = []
    if i != 0 and map_plan[i - 1, j]:
        neighbours.append((i - 1, j))
    if i != shape[0] - 1 and map_plan[i + 1, j]:
        neighbours.append((i + 1, j))
    if j != 0 and map_plan[i, j - 1]:
        neighbours.append((i, j - 1))
    if j != shape[1] - 1 and map_plan[i, j + 1]:
        neighbours.append((i, j + 1))

    return neighbours


def getIslands(map_plan: np.ndarray, min_island_size: int) -> List[Island]:
    """
    counts the number of islands in a bool numpy array
    example with 3 islands (with sizes 6, 1, 3):
    True    False   True    False
    True    True    False   True
    True    False   False   True
    True    True    False   True
    """
    shape: Tuple[int, int] = map_plan.shape[0], map_plan.shape[1]
    islands: List[Island] = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            # explore island
            if map_plan[i, j]:
                map_plan[i, j] = False
                temp_island: Island = Island(i, j)
                stack: List[Tuple[int, int]] = [(i, j)]
                while len(stack) > 0:
                    cur: Tuple[int, int] = stack.pop()
                    for neighbour in getNeighbours(map_plan, shape, cur[0], cur[1]):
                        map_plan[neighbour[0], neighbour[1]] = False
                        stack.append(neighbour)
                        temp_island.extendIsland(neighbour[0], neighbour[1])
                if temp_island.size >= min_island_size:
                    islands.append(temp_island)

    return islands


def getBoxes(
    img: np.ndarray,
    r_min: int,
    r_max: int,
    g_min: int,
    g_max: int,
    b_min: int,
    b_max: int,
    min_island_size: int = 100,
) -> List[Tuple[int, int, int, int]]:
    map_plan: np.ndarray = np.ones((img.shape[0], img.shape[1]), dtype=bool)
    map_plan[
        (
            ((img[:, :, 0] < r_min) | (img[:, :, 0] > r_max))
            | ((img[:, :, 1] < g_min) | (img[:, :, 1] > g_max))
            | ((img[:, :, 2] < b_min) | (img[:, :, 2] > b_max))
        )
    ] = False

    islands: List[Island] = getIslands(map_plan, min_island_size)
    return [
        (elem.min_row, elem.min_col, elem.max_row + 1, elem.max_col + 1)
        for elem in islands
    ]
