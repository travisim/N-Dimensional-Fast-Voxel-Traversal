import numpy as np
import itertools
from typing import List, Tuple, Optional, Dict, Any, Callable

def round2(x):
    """
    Rounds a float, int, or numpy array to 2 decimal places.
    """
    if isinstance(x, (float, int)):
        return round(x, 2)
    elif isinstance(x, np.ndarray):
        return np.round(x, 2)
    elif isinstance(x, list):
        return [round2(i) for i in x]
    else:
        return x

class RayTracerBase:
    """
    Base N-Dimensional Ray Tracer.
    Handles the core ray tracing logic without obstacles or visualization.
    """
    def __init__(self):
        self.x_0: Optional[np.ndarray] = None
        self.x_f: Optional[np.ndarray] = None
        self.delta_x: Optional[np.ndarray] = None
        self.abs_delta_x: Optional[np.ndarray] = None
        self.norm_delta_x: Optional[float] = None
        self.delta_x_sign: Optional[np.ndarray] = None
        self.k: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.D_0: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.l: float = 0
        self.t: int = 0
        self.n: int = 0

    def init(self, x_0: np.ndarray, x_f: np.ndarray):
        """
        Initializes the ray from start (x_0) to goal (x_f).
        """
        self.x_0 = np.array(x_0, dtype=float)
        self.x_f = np.array(x_f, dtype=float)
        self.n = len(x_0)
        self.t = 0
        
        self.delta_x = self.x_f - self.x_0
        self.abs_delta_x = np.abs(self.delta_x)
        self.delta_x_sign = np.array([self._sign(dx) for dx in self.delta_x])
        self.l = 0
        self.k = np.zeros(self.n, dtype=int)
        self.norm_delta_x = np.linalg.norm(self.delta_x)
        self.y = self._floor_ceil_conditional(self.x_0, -self.delta_x).astype(int)
        
        self._determine_front_cells()

        self.D = np.zeros(self.n)
        for i in range(self.n):
            if self.delta_x[i] < 0:
                self.D[i] = (np.floor(self.x_0[i]) - self.x_0[i]) / self.delta_x[i]
            elif abs(self.delta_x[i]) < 1e-10:
                self.D[i] = float('inf')
            else:
                self.D[i] = (np.ceil(self.x_0[i]) - self.x_0[i]) / self.delta_x[i]

            if abs(self.D[i]) < 1e-9 and abs(self.delta_x[i]) > 1e-9:
                self.D[i] = 1.0 / abs(self.delta_x[i])
        self.D_0 = self.D.copy()

    def _floor_ceil_conditional(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.zeros_like(a)
        for i in range(len(a)):
            if b[i] <= 0:
                result[i] = np.floor(a[i])
            else:
                result[i] = np.ceil(a[i])
        return result
    
    def _sign(self, x: float) -> int:
        if x > 0: return 1
        elif x == 0: return 0
        else: return -1

    def _determine_front_cells(self):
        self.F = np.array(self._determine_front_cells_recursive(0, np.zeros(self.n, dtype=int)))

    def _determine_front_cells_recursive(self, dim, current_f):
        if dim == self.n:
            return [current_f.copy()]

        f_list = []
        is_delta_x_zero = abs(self.delta_x[dim]) < 1e-10
        is_x0_integer = abs(self.x_0[dim] - round(self.x_0[dim])) < 1e-10

        if is_delta_x_zero and is_x0_integer:
            current_f[dim] = -1
            f_list.extend(self._determine_front_cells_recursive(dim + 1, current_f))
            
            current_f[dim] = 0
            f_list.extend(self._determine_front_cells_recursive(dim + 1, current_f))
        else:
            if self.delta_x_sign[dim] < 0:
                current_f[dim] = -1
            else:
                current_f[dim] = 0
            f_list.extend(self._determine_front_cells_recursive(dim + 1, current_f))
        return f_list

    def coords(self) -> np.ndarray:
        if self.norm_delta_x == 0:
            return self.x_0.copy()
        return self.x_0 + (self.l / self.norm_delta_x) * self.delta_x
    
    def front_cells(self) -> List[np.ndarray]:
        cells = []
        for f_j in self.F:
            cell_coord = self.y + f_j
            cells.append(cell_coord)
        return cells
    
    def length(self) -> float:
        return self.l
    
    def reached(self) -> bool:
        return np.min(self.D) >= 1.0

    def next(self):
        min_D_value = np.min(self.D)
        i_star_indices = np.where(self.D == min_D_value)[0]
        
        for i_star in i_star_indices:
            self.k[i_star] += 1
        
        self.l = min_D_value * self.norm_delta_x
    
        for i_star in i_star_indices:
            if abs(self.delta_x[i_star]) > 1e-10:
                self.D[i_star] = self.D_0[i_star] + (self.k[i_star] / abs(self.delta_x[i_star]))
            else:
                self.D[i_star] = float('inf')

        for i_star in i_star_indices:
            self.y[i_star] += self.delta_x_sign[i_star]

        self.t += 1
        self._determine_front_cells()

class ObstacleSet:
    """
    Manages the set of obstacles.
    """
    def __init__(self, obstacles: Optional[List[np.ndarray]] = None):
        self.obstacle_set = {tuple(obs) for obs in obstacles} if obstacles else set()

    def is_obstacle(self, coord: Tuple[int, ...]) -> bool:
        """Checks if a given coordinate is an obstacle."""
        return tuple(coord) in self.obstacle_set

class ObstacleRayTracer:
    """
    A wrapper for RayTracerBase that handles obstacle detection.
    """
    def __init__(self):
        self.tracer = RayTracerBase()
        self.obstacle_manager: Optional[ObstacleSet] = None
        self.prev_front_cell_status: Optional[np.ndarray] = None
        self.current_front_cell_status: Optional[np.ndarray] = None
        self.loose_dimension: int = 0
        self.possible_offsets: List[np.ndarray] = []

    def init(self, x_0: np.ndarray, x_f: np.ndarray, obstacles: Optional[List[np.ndarray]] = None, loose_dimension: int = 0):
        self.tracer.init(x_0, x_f)
        self.obstacle_manager = ObstacleSet(obstacles)
        self.loose_dimension = loose_dimension
        self.prev_front_cell_status = np.ones(len(self.tracer.F), dtype=int)
        self.possible_offsets = self._generate_loose_dimensions_offsets_from_sign()

    def next(self):
        self.tracer.next()

    def _calculate_search_space(self, all_front_cells: List[np.ndarray]) -> List[Tuple[int, ...]]:
        if not all_front_cells:
            return []
        all_cells = np.vstack(all_front_cells)
        min_coords = np.min(all_cells, axis=0).astype(int)
        max_coords = np.max(all_cells, axis=0).astype(int)
        ranges = [range(min_c, max_c + 1) for min_c, max_c in zip(min_coords, max_coords)]
        return list(itertools.product(*ranges))

    def _generate_loose_dimensions_offsets_from_sign(self) -> List[np.ndarray]:
        if self.loose_dimension <= 0 or self.loose_dimension > self.tracer.n:
            raise ValueError("loose_dimension must be between 1 and the number of dimensions (inclusive)")

        all_offsets = []
        
        # The dimensions along which the ray is moving
        ray_move_dims = [i for i, d in enumerate(self.tracer.delta_x_sign) if d != 0]

        # Generate offsets for dimensions with movement
        for i in range(1, self.loose_dimension + 1):
            if len(ray_move_dims) < i:
                break
            for dims_to_change in itertools.combinations(ray_move_dims, i):
                offset = np.zeros(self.tracer.n, dtype=int)
                for dim in dims_to_change:
                    offset[dim] = self.tracer.delta_x_sign[dim]
                all_offsets.append(offset)

        # Generate offsets for dimensions without movement
        non_move_dims = [i for i, d in enumerate(self.tracer.delta_x_sign) if d == 0]
        for i in range(1, self.loose_dimension + 1):
            if len(non_move_dims) < i:
                break
            for dims_to_change in itertools.combinations(non_move_dims, i):
                # For each combination, generate all possible sign combinations
                for signs in itertools.product([-1, 1], repeat=i):
                    offset = np.zeros(self.tracer.n, dtype=int)
                    for j, dim in enumerate(dims_to_change):
                        offset[dim] = signs[j]
                    all_offsets.append(offset)
        
        # Also handle mixed offsets
        if self.loose_dimension > 1:
            for i in range(1, self.loose_dimension):
                # i = number of moving dimensions
                # self.loose_dimension - i = number of non-moving dimensions
                
                num_non_move_dims = self.loose_dimension - i
                if len(ray_move_dims) < i or len(non_move_dims) < num_non_move_dims:
                    continue

                for move_dims_combo in itertools.combinations(ray_move_dims, i):
                    for non_move_dims_combo in itertools.combinations(non_move_dims, num_non_move_dims):
                        for signs in itertools.product([-1, 1], repeat=num_non_move_dims):
                            offset = np.zeros(self.tracer.n, dtype=int)
                            for dim in move_dims_combo:
                                offset[dim] = self.tracer.delta_x_sign[dim]
                            for j, dim in enumerate(non_move_dims_combo):
                                offset[dim] = signs[j]
                            all_offsets.append(offset)

        return all_offsets

    def _dfs_get_traced_cells(self, start_cell: tuple, search_space: set, is_obstacle_func: Callable[[Tuple[int, ...]], bool]) -> set:
        if is_obstacle_func(start_cell):
            return set()
            
        stack = [start_cell]
        visited = {start_cell}

        while stack:
            current_cell = stack.pop()

            for offset in self.possible_offsets:
                neighbor = tuple(np.array(current_cell) + offset)

                if neighbor in search_space and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        
        return visited

    def is_hit_obstacle(self, prev_front_cells: List[np.ndarray], current_front_cells: List[np.ndarray], is_obstacle_func: Callable[[Tuple[int, ...]], bool]) -> bool:
        obstacle_set = {cell for cell in self._calculate_search_space(prev_front_cells + current_front_cells) if is_obstacle_func(cell)}
        search_space = {cell for cell in self._calculate_search_space(prev_front_cells + current_front_cells) if not is_obstacle_func(cell)}

        _prev_front_cells_list = [tuple(c) for c in prev_front_cells]
        _current_front_cells_list = [tuple(c) for c in current_front_cells]

        self.current_front_cell_status = np.zeros(len(_current_front_cells_list), dtype=int)

        for i, start_cell in enumerate(_prev_front_cells_list):
            if self.prev_front_cell_status[i] == 1:
                traced_cells = self._dfs_get_traced_cells(start_cell, search_space, is_obstacle_func)
                
                for j, c_cell in enumerate(_current_front_cells_list):
                    if c_cell in traced_cells:
                        self.current_front_cell_status[j] = 1
        
        obstacle_hit = np.sum(self.current_front_cell_status) == 0
        if not obstacle_hit:
            self.prev_front_cell_status = self.current_front_cell_status.copy()
        return obstacle_hit


class RayTracerVisualizer:
    """
    Handles visualization of the ray tracing process.
    """
    def __init__(self):
        self.tracer = ObstacleRayTracer()

    def traverse(self, x_0: np.ndarray, x_f: np.ndarray, obstacles: Optional[List[np.ndarray]] = None, loose_dimension: int = 0):
        self.tracer.init(x_0, x_f, obstacles, loose_dimension)
        
        path = [self.tracer.tracer.x_0.copy()]
        all_front_cells = [self.tracer.tracer.front_cells()]
        y_coords_history = [self.tracer.tracer.y.copy()]
        intersection_coords = [self.tracer.tracer.x_0.copy()]

        isGoalReached = False
        obstacle_hit = False

        initial_front_cells = all_front_cells[-1]
        if self.tracer.is_hit_obstacle(initial_front_cells, initial_front_cells, self.tracer.obstacle_manager.is_obstacle):
            obstacle_hit = True
        elif np.array_equal(self.tracer.tracer.x_0, x_f):
            isGoalReached = True
            if self.tracer.obstacle_manager.is_obstacle(tuple(np.round(x_f).astype(int))):
                 obstacle_hit = True
                 isGoalReached = False
        else:
            while not self.tracer.tracer.reached():
                prev_front_cells = all_front_cells[-1]
                self.tracer.next()
                new_front_cells = self.tracer.tracer.front_cells()
                
                path.append(self.tracer.tracer.coords().copy())
                intersection_coords.append(self.tracer.tracer.coords().copy())
                y_coords_history.append(self.tracer.tracer.y.copy())
                
                if self.tracer.is_hit_obstacle(prev_front_cells, new_front_cells, self.tracer.obstacle_manager.is_obstacle):
                    obstacle_hit = True
                    break
                
                all_front_cells.append(new_front_cells)

            if not obstacle_hit:
                if not path or not np.array_equal(path[-1], x_f):
                    path.append(x_f.copy())
                isGoalReached = True
        
        return path, all_front_cells, intersection_coords, y_coords_history, obstacle_hit, isGoalReached
