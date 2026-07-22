from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from actions import Action
from robot import KNOWN_WALL, KNOWN_FREE, UNKNOWN, Position, Robot


    
    

# Broken logic, this is the same as random
def move_toward_unknown(robot: Robot, rng: np.random.Generator) -> Action:
        # Move toward unknown cells if possible
        if not robot.alive:
            raise RuntimeError("Inactive robot cannot move")
        row, col = robot.pos
        possible_actions = [
            (0, 0),   # Stay in place
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1),   # Right
        ]
        valid_actions = [
            (dr, dc) for dr, dc in possible_actions
            if 0 <= row + dr < robot.map_shape[0]
            and 0 <= col + dc < robot.map_shape[1]
            and robot.belief_map[row + dr, col + dc] != KNOWN_WALL
        ]
        if not valid_actions:
            return (0, 0)  # No valid moves, stay in place

        # Prioritize actions that lead to unknown cells
        unknown_actions = [
            (dr, dc) for dr, dc in valid_actions
            if robot.belief_map[row + dr, col + dc] == UNKNOWN
        ]
        if unknown_actions:
            return unknown_actions[rng.choice(len(unknown_actions))]
        return valid_actions[rng.choice(len(valid_actions))]  # No unknown cells, pick randomly
    

def move_toward_unknown_bfs(robot: Robot, rng) -> Action:
    if not robot.alive:
        raise RuntimeError("Inactive robot cannot move")


    @dataclass
    class Node:
        pos: tuple[int, int]
        children: list[Node] = field(default_factory=list)
        parent: Node = None
        
        
        
    def is_frontier_cell(n: Node) -> bool:
        around = [
            (n.pos[0]+1,n.pos[1]),
            (n.pos[0]-1,n.pos[1]),
            (n.pos[0],n.pos[1]+1),
            (n.pos[0],n.pos[1]-1)
            ]
        return any(
            0 <= nr < robot.map_shape[0] and 0 <= nc < robot.map_shape[1] and robot.belief_map[nr, nc] == UNKNOWN
            for nr, nc in around
        )
    
    start = Node(pos=tuple(robot.pos))
    nodes = [start]
    
    queue = deque()
    queue.append(start)
    visited = {start.pos}


    while queue:
        n = queue.popleft()
        possible = [
            (n.pos[0]+1,n.pos[1]),
            (n.pos[0]-1,n.pos[1]),
            (n.pos[0],n.pos[1]+1),
            (n.pos[0],n.pos[1]-1)
        ]
        valid = [(i,j) for (i,j) in possible if (0<=i<robot.map_shape[0] and 0<=j<robot.map_shape[1])]
        for (i,j) in valid:
            if (i,j) in visited:
                continue
            visited.add((i,j))
            if not robot.belief_map[i][j] == KNOWN_WALL:
                
                _n = Node(
                    pos = (i,j),
                    parent = (n)
                )
                queue.append(_n)
                nodes.append(_n)
                
                if robot.belief_map[i][j] == UNKNOWN:
                    temp = _n
                    while not temp.parent is start:
                        temp = temp.parent
                    out: Action = (temp.pos[0] - start.pos[0], temp.pos[1] - start.pos[1])
                    return out
    return (0,0)
        
            
        