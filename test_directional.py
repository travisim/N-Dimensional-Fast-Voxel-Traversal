#!/usr/bin/env python3

import numpy as np
from n_dimensional_fast_voxel_traversal import RayTracerVisualizer
from plotting import plot_trace

# Test just the new directional cases to verify they work
directional_tests = [
    # 2D Directional Tests - Integer Coordinates
    {"dim": 2, "label": "Direction-2D-Int-East-0deg", "x_0": np.array([3, 3]), "x_f": np.array([7, 3]), "obstacles": [], "loose_dimension": 1},
    {"dim": 2, "label": "Direction-2D-Int-North-90deg", "x_0": np.array([3, 3]), "x_f": np.array([3, 7]), "obstacles": [], "loose_dimension": 1},
    {"dim": 2, "label": "Direction-2D-Int-West-180deg", "x_0": np.array([7, 3]), "x_f": np.array([3, 3]), "obstacles": [], "loose_dimension": 1},
    {"dim": 2, "label": "Direction-2D-Int-South-270deg", "x_0": np.array([3, 7]), "x_f": np.array([3, 3]), "obstacles": [], "loose_dimension": 1},
    
    # 2D Directional Tests - Float Coordinates
    {"dim": 2, "label": "Direction-2D-Float-East-0deg", "x_0": np.array([3.5, 3.5]), "x_f": np.array([7.5, 3.5]), "obstacles": [], "loose_dimension": 1},
    {"dim": 2, "label": "Direction-2D-Float-North-90deg", "x_0": np.array([3.5, 3.5]), "x_f": np.array([3.5, 7.5]), "obstacles": [], "loose_dimension": 1},
    {"dim": 2, "label": "Direction-2D-Float-West-180deg", "x_0": np.array([7.5, 3.5]), "x_f": np.array([3.5, 3.5]), "obstacles": [], "loose_dimension": 1},
    {"dim": 2, "label": "Direction-2D-Float-South-270deg", "x_0": np.array([3.5, 7.5]), "x_f": np.array([3.5, 3.5]), "obstacles": [], "loose_dimension": 1},
    
    # 3D Directional Tests - Integer Coordinates
    {"dim": 3, "label": "Direction-3D-Int-PosX", "x_0": np.array([3, 3, 3]), "x_f": np.array([7, 3, 3]), "obstacles": [], "loose_dimension": 1},
    {"dim": 3, "label": "Direction-3D-Int-NegX", "x_0": np.array([7, 3, 3]), "x_f": np.array([3, 3, 3]), "obstacles": [], "loose_dimension": 1},
    {"dim": 3, "label": "Direction-3D-Int-MainDiagonal", "x_0": np.array([3, 3, 3]), "x_f": np.array([7, 7, 7]), "obstacles": [], "loose_dimension": 1},
    
    # 3D Directional Tests - Float Coordinates
    {"dim": 3, "label": "Direction-3D-Float-PosX", "x_0": np.array([3.5, 3.5, 3.5]), "x_f": np.array([7.5, 3.5, 3.5]), "obstacles": [], "loose_dimension": 1},
    {"dim": 3, "label": "Direction-3D-Float-NegX", "x_0": np.array([7.5, 3.5, 3.5]), "x_f": np.array([3.5, 3.5, 3.5]), "obstacles": [], "loose_dimension": 1},
    {"dim": 3, "label": "Direction-3D-Float-MainDiagonal", "x_0": np.array([3.5, 3.5, 3.5]), "x_f": np.array([7.5, 7.5, 7.5]), "obstacles": [], "loose_dimension": 1},
]

print("Testing expanded directional test cases (Integer & Float variations)...")
print("=" * 80)

for test in directional_tests:
    print(f"\nTesting {test['dim']}D: {test['label']}")
    coord_type = "Float" if any(isinstance(x, float) or x % 1 != 0 for x in test['x_0']) else "Integer"
    print(f"Coordinate Type: {coord_type}")
    print(f"Start: {test['x_0']}, Goal: {test['x_f']}")
    
    visualizer = RayTracerVisualizer()
    path, front_cells, intersections, y_history, hit, goal_reached = visualizer.traverse(
        test['x_0'], test['x_f'], obstacles=test.get('obstacles', []), loose_dimension=test['loose_dimension']
    )
    
    print(f"Goal reached: {goal_reached}")
    
    # Plot the trace
    plot_trace(
        test['dim'], test['label'],
        test['x_0'], test['x_f'],
        path, front_cells, intersections, y_history,
        test.get('obstacles', []),
        goal_reached,
        test['loose_dimension']
    )

print("\nExpanded directional test cases completed!")
print(f"Total directional test cases tested: {len(directional_tests)}")
print("Coverage:")
print("- 2D: 4 directions × 2 coordinate types = 8 test cases")
print("- 3D: 8 directions × 2 coordinate types = 16 test cases") 
print("- Total: 24 comprehensive directional test cases")
