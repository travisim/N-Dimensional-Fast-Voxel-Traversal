#!/usr/bin/env python3

import numpy as np
from n_dimensional_fast_voxel_traversal import RayTracerVisualizer
from plotting import plot_trace

# Test a few specific cases to verify viewport scaling
test_cases = [
    # Small coordinates
    {"dim": 2, "label": "Small-2D", "x_0": np.array([1, 1]), "x_f": np.array([3, 3]), "obstacles": []},
    
    # Large coordinates - this was the problematic case
    {"dim": 2, "label": "Large-2D", "x_0": np.array([100, 100]), "x_f": np.array([105, 105]), "obstacles": []},
    
    # Float coordinates with odd ranges
    {"dim": 2, "label": "Float-2D", "x_0": np.array([1.2, 1.8]), "x_f": np.array([5.7, 6.3]), "obstacles": []},
    
    # 3D large coordinates
    {"dim": 3, "label": "Large-3D", "x_0": np.array([10, 10, 10]), "x_f": np.array([15, 15, 15]), "obstacles": [np.array([12, 12, 12])]},
    
    # Negative coordinates
    {"dim": 2, "label": "Negative-2D", "x_0": np.array([-2, -2]), "x_f": np.array([2, 2]), "obstacles": []},
]

print("Testing viewport scaling with integer boundaries...")
print("=" * 60)

for test in test_cases:
    print(f"\nTesting {test['dim']}D: {test['label']}")
    print(f"Start: {test['x_0']}, Goal: {test['x_f']}")
    
    visualizer = RayTracerVisualizer()
    path, front_cells, intersections, y_history, hit, goal_reached = visualizer.traverse(
        test['x_0'], test['x_f'], obstacles=test.get('obstacles', []), loose_dimension=1
    )
    
    print(f"Goal reached: {goal_reached}")
    
    # Plot the trace
    plot_trace(
        test['dim'], test['label'],
        test['x_0'], test['x_f'],
        path, front_cells, intersections, y_history,
        test.get('obstacles', []),
        goal_reached,
        1  # loose_dimension
    )

print("\nViewport scaling test completed!")
