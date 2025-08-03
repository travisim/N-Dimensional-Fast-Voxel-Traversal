import numpy as np
import itertools
from typing import List, Tuple, Optional, Dict, Any, Callable

def round2(x):
    if isinstance(x, (float, int)):
        return round(x, 2)
    elif isinstance(x, np.ndarray):
        return np.round(x, 2)
    elif isinstance(x, list):
        return [round2(i) for i in x]
    else:
        return x

class RayTracerBase:
    def __init__(self):
        self._initial_corner_coords: Optional[np.ndarray] = None
        self._final_corner_coords: Optional[np.ndarray] = None
        self._direction: Optional[np.ndarray] = None
        self._abs_direction: Optional[np.ndarray] = None
        self._total_dist: Optional[float] = None
        self._sign_direction: Optional[np.ndarray] = None
        self._fractional_dist_counters: Optional[np.ndarray] = None
        self._next_fractional_dist: Optional[np.ndarray] = None
        self._initial_fractional_dist: Optional[np.ndarray] = None
        self._current_corner_coords: Optional[np.ndarray] = None
        self._adjacent_center_coords: Optional[np.ndarray] = None
        self._current_fractional_dist: float = 0
        self.t: int = 0
        self.n: int = 0

    def init(self, x_0: np.ndarray, x_f: np.ndarray):
        self._initial_corner_coords = np.array(x_0, dtype=float)
        self._final_corner_coords = np.array(x_f, dtype=float)
        self.n = len(x_0)
        self.t = 0
        
        self._direction = self._final_corner_coords - self._initial_corner_coords
        self._abs_direction = np.abs(self._direction)
        self._current_fractional_dist = 0
        self._fractional_dist_counters = np.zeros(self.n, dtype=int)
        self._total_dist = np.linalg.norm(self._direction)
        
        self._sign_direction = np.zeros(self.n, dtype=int)
        self._current_corner_coords = np.zeros(self.n, dtype=int)
        self._next_fractional_dist = np.zeros(self.n)
        self._initial_fractional_dist = np.zeros(self.n)
        
        THRES = 1e-10
        
        for i in range(self.n):
            if self._direction[i] > THRES:
                self._sign_direction[i] = 1
                self._current_corner_coords[i] = int(np.floor(self._initial_corner_coords[i]))
                self._initial_fractional_dist[i] = (self._current_corner_coords[i] - self._initial_corner_coords[i] + self._sign_direction[i]) / self._direction[i]
                self._next_fractional_dist[i] = self._initial_fractional_dist[i]
            elif self._direction[i] < -THRES:
                self._sign_direction[i] = -1
                self._current_corner_coords[i] = int(np.ceil(self._initial_corner_coords[i]))
                self._initial_fractional_dist[i] = (self._current_corner_coords[i] - self._initial_corner_coords[i] + self._sign_direction[i]) / self._direction[i]
                self._next_fractional_dist[i] = self._initial_fractional_dist[i]
            else:
                self._sign_direction[i] = 0
                self._initial_fractional_dist[i] = float('inf')
                self._next_fractional_dist[i] = float('inf')
                self._current_corner_coords[i] = int(np.floor(self._initial_corner_coords[i]))
        
        self._determine_front_cells()

    def _determine_front_cells(self):
        self._adjacent_center_coords = np.array(self._determine_front_cells_recursive(0, np.zeros(self.n, dtype=int)))

    def _determine_front_cells_recursive(self, dim, current_f):
        if dim == self.n:
            return [current_f.copy()]

        f_list = []
        THRES = 1e-10
        
        is_delta_x_zero = abs(self._direction[dim]) <= THRES
        is_x0_integer = abs(self._initial_corner_coords[dim] - round(self._initial_corner_coords[dim])) < 1e-10

        if is_delta_x_zero and is_x0_integer:
            current_f[dim] = -1
            f_list.extend(self._determine_front_cells_recursive(dim + 1, current_f))
            
            current_f[dim] = 0
            f_list.extend(self._determine_front_cells_recursive(dim + 1, current_f))
        else:
            if self._sign_direction[dim] < 0:
                current_f[dim] = -1
            else:
                current_f[dim] = 0
            f_list.extend(self._determine_front_cells_recursive(dim + 1, current_f))
        return f_list

    def coords(self) -> np.ndarray:
        if self._total_dist == 0:
            return self._initial_corner_coords.copy()
        return self._initial_corner_coords + (self._current_fractional_dist / self._total_dist) * self._direction
    
    def front_cells(self) -> List[np.ndarray]:
        cells = []
        for f_j in self._adjacent_center_coords:
            cell_coord = self._current_corner_coords + f_j
            cells.append(cell_coord)
        return cells
    
    def length(self) -> float:
        return self._current_fractional_dist
    
    def reached(self) -> bool:
        return np.min(self._next_fractional_dist) >= 1.0

    def next(self):
        min_D_value = np.min(self._next_fractional_dist)
        i_star_indices = np.where(self._next_fractional_dist == min_D_value)[0]
        
        for i_star in i_star_indices:
            self._fractional_dist_counters[i_star] += 1
        
        self._current_fractional_dist = min_D_value * self._total_dist
    
        for i_star in i_star_indices:
            if abs(self._direction[i_star]) > 1e-10:
                self._next_fractional_dist[i_star] = self._initial_fractional_dist[i_star] + (self._fractional_dist_counters[i_star] / abs(self._direction[i_star]))
            else:
                self._next_fractional_dist[i_star] = float('inf')

        for i_star in i_star_indices:
            self._current_corner_coords[i_star] += self._sign_direction[i_star]

     

class RayTracer:
    def __init__(self):
        self._tracer: Optional[Any] = None

    def init(self, x_0: np.ndarray, x_f: np.ndarray):
        x_0_arr = np.asarray(x_0)
        x_f_arr = np.asarray(x_f)
       
        self._tracer = RayTracerBase()
        
        self._tracer.init(x_0_arr, x_f_arr)

    def __getattr__(self, name: str) -> Any:
        if self._tracer is None:
            raise AttributeError("RayTracer has not been initialized. Call init() first.")
        return getattr(self._tracer, name)

class ObstacleSet:
    def __init__(self, obstacles: Optional[List[np.ndarray]] = None):
        self.obstacle_set = {tuple(obs) for obs in obstacles} if obstacles else set()

    def is_obstacle(self, coord: Tuple[int, ...]) -> bool:
        return tuple(coord) in self.obstacle_set

def create_obstacle_function(obstacles: Optional[List[np.ndarray]] = None) -> Callable[[Tuple[int, ...]], bool]:
    if obstacles is None:
        return lambda coord: False
    
    obstacle_set = {tuple(obs.astype(int)) for obs in obstacles}
    return lambda coord: tuple(coord) in obstacle_set


class ObstacleRayTracer:
    def __init__(self):
        self.tracer = RayTracer()
        self.is_obstacle: Callable[[Tuple[int, ...]], bool] = lambda coord: False
        self.prev_front_cell_status: Optional[np.ndarray] = None
        self.current_front_cell_status: Optional[np.ndarray] = None
        self.loose_dimension: int = 0
        self.possible_offsets: List[np.ndarray] = []

    def init(self, x_0: np.ndarray, x_f: np.ndarray, is_obstacle: Optional[Callable[[Tuple[int, ...]], bool]] = None, loose_dimension: int = 0):
        self.tracer.init(x_0, x_f)
        self.is_obstacle = is_obstacle or (lambda coord: False)
        self.loose_dimension = loose_dimension
        self.prev_front_cell_status = np.ones(len(self.tracer._adjacent_center_coords), dtype=int)
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
        
        ray_move_dims = [i for i, d in enumerate(self.tracer._sign_direction) if d != 0]

        for i in range(1, self.loose_dimension + 1):
            if len(ray_move_dims) < i:
                break
            for dims_to_change in itertools.combinations(ray_move_dims, i):
                offset = np.zeros(self.tracer.n, dtype=int)
                for dim in dims_to_change:
                    offset[dim] = self.tracer._sign_direction[dim]
                all_offsets.append(offset)

        non_move_dims = [i for i, d in enumerate(self.tracer._sign_direction) if d == 0]
        for i in range(1, self.loose_dimension + 1):
            if len(non_move_dims) < i:
                break
            for dims_to_change in itertools.combinations(non_move_dims, i):
                for signs in itertools.product([-1, 1], repeat=i):
                    offset = np.zeros(self.tracer.n, dtype=int)
                    for j, dim in enumerate(dims_to_change):
                        offset[dim] = signs[j]
                    all_offsets.append(offset)
        
        if self.loose_dimension > 1:
            for i in range(1, self.loose_dimension):
                num_non_move_dims = self.loose_dimension - i
                if len(ray_move_dims) < i or len(non_move_dims) < num_non_move_dims:
                    continue

                for move_dims_combo in itertools.combinations(ray_move_dims, i):
                    for non_move_dims_combo in itertools.combinations(non_move_dims, num_non_move_dims):
                        for signs in itertools.product([-1, 1], repeat=num_non_move_dims):
                            offset = np.zeros(self.tracer.n, dtype=int)
                            for dim in move_dims_combo:
                                offset[dim] = self.tracer._sign_direction[dim]
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
    def __init__(self):
        self.tracer = ObstacleRayTracer()

    def traverse(self, x_0: np.ndarray, x_f: np.ndarray, obstacles: Optional[List[np.ndarray]] = None, loose_dimension: int = 0, is_obstacle: Optional[Callable[[Tuple[int, ...]], bool]] = None):
        if is_obstacle is None and obstacles is not None:
            is_obstacle = create_obstacle_function(obstacles)
        elif is_obstacle is None:
            is_obstacle = lambda coord: False
        
        self.tracer.init(x_0, x_f, is_obstacle, loose_dimension)
        
        path = [self.tracer.tracer._initial_corner_coords.copy()]
        all_front_cells = [self.tracer.tracer.front_cells()]
        y_coords_history = [self.tracer.tracer._current_corner_coords.copy()]
        intersection_coords = [self.tracer.tracer._initial_corner_coords.copy()]

        isGoalReached = False
        obstacle_hit = False

        initial_front_cells = all_front_cells[-1]
        if self.tracer.is_hit_obstacle(initial_front_cells, initial_front_cells, self.tracer.is_obstacle):
            obstacle_hit = True
        elif np.array_equal(self.tracer.tracer._initial_corner_coords, x_f):
            isGoalReached = True
            if self.tracer.is_obstacle(tuple(np.round(x_f).astype(int))):
                 obstacle_hit = True
                 isGoalReached = False
        else:
            while not self.tracer.tracer.reached():
                prev_front_cells = all_front_cells[-1]
                self.tracer.next()
                new_front_cells = self.tracer.tracer.front_cells()
                
                path.append(self.tracer.tracer.coords().copy())
                intersection_coords.append(self.tracer.tracer.coords().copy())
                y_coords_history.append(self.tracer.tracer._current_corner_coords.copy())
                
                if self.tracer.is_hit_obstacle(prev_front_cells, new_front_cells, self.tracer.is_obstacle):
                    obstacle_hit = True
                    break
                
                all_front_cells.append(new_front_cells)

            if not obstacle_hit:
                if not path or not np.array_equal(path[-1], x_f):
                    path.append(x_f.copy())
                isGoalReached = True
        
        return path, all_front_cells, intersection_coords, y_coords_history, obstacle_hit, isGoalReached
