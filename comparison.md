# Comparison: `RayTracerBase` vs. `RayTracerBaseIntegerOptimized`

This document outlines the key differences between the `RayTracerBase` and `RayTracerBaseIntegerOptimized` classes. The `RayTracerBaseIntegerOptimized` class is a specialized version of `RayTracerBase` designed to be more efficient when both the start and end coordinates of the ray are integers.

## Key Differences

| Feature | `RayTracerBase` | `RayTracerBaseIntegerOptimized` |
| :--- | :--- | :--- |
| **Input Data Type** | Handles `float` and `integer` coordinates. All inputs are cast to `float`. | Optimized specifically for `integer` coordinates. Inputs are cast to `int`. |
| **Coordinate Initialization (`y`)** | Uses a conditional `floor` or `ceil` operation based on the direction of the ray (`delta_x`) to determine the initial voxel coordinates. This is necessary to correctly handle floating-point starting positions. | Directly uses the integer starting coordinates (`x_0`) as the initial voxel coordinates (`y`). This is simpler and faster as no rounding is needed. |
| **Distance Calculation (`D`)** | Calculates the initial distance `D` to the next voxel boundary based on the fractional part of the starting coordinates. This involves division and conditional logic to handle different ray directions. | Calculates `D` as the inverse of the absolute delta (`1 / abs(delta_x)`). This is possible because for integer coordinates, the distance to the next voxel boundary is always 1 unit along that axis. |
| **Front Cell Determination** | The `_determine_front_cells_recursive` method includes checks for whether `delta_x` is close to zero and whether `x_0` is an integer to handle edge cases at grid lines. | The `_determine_front_cells_recursive` method is simplified. It only checks if `delta_x` is exactly zero, as the starting coordinates are guaranteed to be integers. |

## Code Snippets

### 1. Data Type and Initial Voxel Coordinate (`y`)

**`RayTracerBase`:**
```python
def init(self, x_0: np.ndarray, x_f: np.ndarray):
    self.x_0 = np.array(x_0, dtype=float)
    # ...
    self.y = self._floor_ceil_conditional(self.x_0, -self.delta_x).astype(int)

def _floor_ceil_conditional(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.zeros_like(a)
    for i in range(len(a)):
        if b[i] <= 0:
            result[i] = np.floor(a[i])
        else:
            result[i] = np.ceil(a[i])
    return result
```

**`RayTracerBaseIntegerOptimized`:**
```python
def init(self, x_0: np.ndarray, x_f: np.ndarray):
    self.x_0 = np.array(x_0, dtype=int)
    # ...
    self.y = self.x_0.copy()
```

### 2. Distance Calculation (`D`)

**`RayTracerBase`:**
```python
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
```

**`RayTracerBaseIntegerOptimized`:**
```python
self.D = np.zeros(self.n, dtype=float)
for i in range(self.n):
    if self.abs_delta_x[i] == 0:
        self.D[i] = float('inf')
    else:
        self.D[i] = 1.0 / self.abs_delta_x[i]
```

### 3. Front Cell Determination

**`RayTracerBase`:**
```python
def _determine_front_cells_recursive(self, dim, current_f):
    # ...
    is_delta_x_zero = abs(self.delta_x[dim]) < 1e-10
    is_x0_integer = abs(self.x_0[dim] - round(self.x_0[dim])) < 1e-10

    if is_delta_x_zero and is_x0_integer:
        # ...
```

**`RayTracerBaseIntegerOptimized`:**
```python
def _determine_front_cells_recursive(self, dim, current_f):
    # ...
    is_delta_x_zero = self.delta_x[dim] == 0

    if is_delta_x_zero:
        # ...
```

## Summary of Optimizations

The optimizations in `RayTracerBaseIntegerOptimized` stem from the assumption that the grid and the ray's start/end points are perfectly aligned, which is true when dealing with integer coordinates.

1.  **No Floating-Point Inaccuracy:** By working exclusively with integers, it avoids potential floating-point precision issues.
2.  **Simplified Calculations:** The logic for determining the initial voxel and the distance to the next voxel boundary is significantly simpler, removing conditional branches and complex calculations.
3.  **Performance:** The reduction in calculations and branching leads to a more performant ray tracing implementation for the specific case of integer coordinates.

The `RayTracer` factory class automatically selects `RayTracerBaseIntegerOptimized` when it detects integer inputs, ensuring that these optimizations are applied transparently without requiring manual selection.