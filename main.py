import numpy as np
from n_dimensional_fast_voxel_traversal import RayTracerVisualizer
from plotting import plot_trace

if __name__ == "__main__":
    # Comprehensive Test Suite
    tests = [
        # 2D Test Cases
        # {"dim": 2, "label": "Start Integer, Goal Float", "x_0": np.array([1, 1]), "x_f": np.array([4.5, 4.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Start Float, Goal Integer", "x_0": np.array([1.5, 1.5]), "x_f": np.array([4, 4]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Both Float", "x_0": np.array([1.5, 1.5]), "x_f": np.array([4.5, 4.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Both Float (different values)", "x_0": np.array([1.2, 1.8]), "x_f": np.array([5.7, 6.3]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Both Integer", "x_0": np.array([1, 1]), "x_f": np.array([4, 4]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Vertical", "x_0": np.array([2, 1]), "x_f": np.array([2, 5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Horizontal", "x_0": np.array([1, 2]), "x_f": np.array([5, 2]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Diagonal", "x_0": np.array([1, 1]), "x_f": np.array([5, 5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "45-degree", "x_0": np.array([1, 1]), "x_f": np.array([5, 5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "End at Goal which is Obstacle", "x_0": np.array([1, 1]), "x_f": np.array([3, 3]), "obstacles": [np.array([3, 3])], "loose_dimension": 1},
        # {"dim": 2, "label": "Start and Goal is Same", "x_0": np.array([2, 2]), "x_f": np.array([2, 2]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Start and Goal is Same but at Obstacle", "x_0": np.array([2, 2]), "x_f": np.array([2, 2]), "obstacles": [np.array([2, 2])], "loose_dimension": 1},
        # {"dim": 2, "label": "Start at Obstacle", "x_0": np.array([1, 1]), "x_f": np.array([5, 5]), "obstacles": [np.array([1, 1])], "loose_dimension": 1},
        # {"dim": 2, "label": "Start at Obstacle", "x_0": np.array([0, 0]), "x_f": np.array([4, 4]), "obstacles": [np.array([0, 1]),np.array([1, 0])], "loose_dimension": 1},

        # # 2D Cardinal Directions
        # {"dim": 2, "label": "North", "x_0": np.array([2.5, 1.5]), "x_f": np.array([2.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "South", "x_0": np.array([2.5, 5.5]), "x_f": np.array([2.5, 1.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "East", "x_0": np.array([1.5, 2.5]), "x_f": np.array([5.5, 2.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "West", "x_0": np.array([5.5, 2.5]), "x_f": np.array([1.5, 2.5]), "obstacles": [], "loose_dimension": 1},
        # # 2D Quadrants
        # {"dim": 2, "label": "Quadrant 1 (NE)", "x_0": np.array([1.5, 1.5]), "x_f": np.array([5.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Quadrant 2 (NW)", "x_0": np.array([5.5, 1.5]), "x_f": np.array([1.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Quadrant 3 (SW)", "x_0": np.array([5.5, 5.5]), "x_f": np.array([1.5, 1.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 2, "label": "Quadrant 4 (SE)", "x_0": np.array([1.5, 5.5]), "x_f": np.array([5.5, 1.5]), "obstacles": [], "loose_dimension": 1},

        # # 3D Test Cases
        # {"dim": 3, "label": "Start Integer, Goal Float", "x_0": np.array([1, 1, 1]), "x_f": np.array([4.5, 4.5, 4.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Start Float, Goal Integer", "x_0": np.array([1.5, 1.5, 1.5]), "x_f": np.array([4, 4, 4]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Both Float", "x_0": np.array([1.5, 1.5, 1.5]), "x_f": np.array([4.5, 4.5, 4.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Both Float (different values)", "x_0": np.array([1.2, 1.8, 1.1]), "x_f": np.array([5.7, 6.3, 5.9]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Both Integer", "x_0": np.array([1, 1, 1]), "x_f": np.array([4, 4, 4]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Vertical", "x_0": np.array([2, 2, 1]), "x_f": np.array([2, 2, 5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Horizontal (along x-axis)", "x_0": np.array([1, 2, 2]), "x_f": np.array([5, 2, 2]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Diagonal", "x_0": np.array([1, 1, 1]), "x_f": np.array([5, 5, 5]), "obstacles": [], "loose_dimension": 2},
        # {"dim": 3, "label": "45-degree (in xy-plane)", "x_0": np.array([1, 1, 1]), "x_f": np.array([5, 5, 1]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "End at Goal which is Obstacle", "x_0": np.array([1, 1, 1]), "x_f": np.array([3, 3, 3]), "obstacles": [np.array([3, 3, 3])], "loose_dimension": 1},
        # {"dim": 3, "label": "Start and Goal is Same", "x_0": np.array([2, 2, 2]), "x_f": np.array([2, 2, 2]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Start and Goal is Same but at Obstacle", "x_0": np.array([2, 2, 2]), "x_f": np.array([2, 2, 2]), "obstacles": [np.array([2, 2, 2])], "loose_dimension": 1},
        # {"dim": 3, "label": "Start at Obstacle", "x_0": np.array([1, 1, 1]), "x_f": np.array([5, 5, 5]), "obstacles": [np.array([1, 1, 1])], "loose_dimension": 1},
        # # 3D Cardinal Directions
        # {"dim": 3, "label": "Up", "x_0": np.array([2.5, 2.5, 1.5]), "x_f": np.array([2.5, 2.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Down", "x_0": np.array([2.5, 2.5, 5.5]), "x_f": np.array([2.5, 2.5, 1.5]), "obstacles": [], "loose_dimension": 1},
        # # 3D Octants
        # {"dim": 3, "label": "Octant 1 (+,+,+)", "x_0": np.array([1.5, 1.5, 1.5]), "x_f": np.array([5.5, 5.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Octant 2 (-,+,+)", "x_0": np.array([5.5, 1.5, 1.5]), "x_f": np.array([1.5, 5.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Octant 3 (-,-,+)", "x_0": np.array([5.5, 5.5, 1.5]), "x_f": np.array([1.5, 1.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Octant 4 (+,-,+)", "x_0": np.array([1.5, 5.5, 1.5]), "x_f": np.array([5.5, 1.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Octant 5 (+,+,-)", "x_0": np.array([1.5, 1.5, 5.5]), "x_f": np.array([5.5, 5.5, 1.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Octant 6 (-,+,-)", "x_0": np.array([5.5, 1.5, 5.5]), "x_f": np.array([1.5, 5.5, 1.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Octant 7 (-,-,-)", "x_0": np.array([5.5, 5.5, 5.5]), "x_f": np.array([1.5, 1.5, 1.5]), "obstacles": [], "loose_dimension": 1},
        # {"dim": 3, "label": "Octant 8 (+,-,-)", "x_0": np.array([1.5, 5.5, 5.5]), "x_f": np.array([5.5, 1.5, 1.5]), "obstacles": [], "loose_dimension": 1},
        
        # Test Loose Dimensions
        {"dim": 2, "label": "Diagonal", "x_0": np.array([1, 1]), "x_f": np.array([5, 5]), "obstacles": [np.array([2, 1]),np.array([1, 2])], "loose_dimension": 1},
        {"dim": 3, "label": "Octant 1 (+,+,+)", "x_0": np.array([1,1,1]), "x_f": np.array([3,3,3]), "obstacles": [np.array([2,1,1]), np.array([1,1,2]), np.array([1, 2, 1])], "loose_dimension": 1},
        {"dim": 3, "label": "Octant 1 (+,+,+)", "x_0": np.array([1,1,1]), "x_f": np.array([3,3,3]), "obstacles": [np.array([2,1,1]), np.array([1,1,2]), np.array([1, 2, 1])], "loose_dimension": 2},
        {"dim": 3, "label": "Octant 1 (+,+,+)", "x_0": np.array([1,1,1]), "x_f": np.array([3,3,3]), "obstacles": [np.array([2,1,1]), np.array([1,1,2]), np.array([1, 2, 1]),np.array([2,1,2]), np.array([1,2,2]), np.array([2, 2, 1])], "loose_dimension": 2},
        {"dim": 3, "label": "Octant 1 (+,+,+)", "x_0": np.array([1,1,1]), "x_f": np.array([3,3,3]), "obstacles": [np.array([2,1,1]), np.array([1,1,2]), np.array([1, 2, 1]),np.array([2,1,2]), np.array([1,2,2]), np.array([2, 2, 1])], "loose_dimension": 3},




  
        {"dim": 3, "label": "Ray along Y-axis with obstacles", "x_0": np.array([2, 0, 3]), "x_f": np.array([2, 5, 3]), "obstacles": [ np.array([1, 2, 3]), np.array([1, 3, 3]),  np.array([2, 2, 2]), np.array([2, 3, 2]), np.array([2, 2, 3]),np.array([1, 3, 2])], "loose_dimension": 2},
        {"dim": 3, "label": "Ray along Y-axis with obstacles", "x_0": np.array([2, 0, 3]), "x_f": np.array([2, 5, 3]), "obstacles": [ np.array([1, 2, 3]), np.array([1, 3, 3]),  np.array([2, 2, 2]), np.array([2, 3, 2]), np.array([2, 2, 3]),np.array([1, 3, 2])], "loose_dimension": 3},
        {"dim": 3, "label": "Ray along Y-axis with obstacles", "x_0": np.array([2, 0, 3]), "x_f": np.array([2, 5, 3]), "obstacles": [ np.array([1, 2, 3]), np.array([1, 3, 3]),  np.array([2, 2, 2]), np.array([2, 3, 2]), np.array([2, 2, 3]),np.array([1, 3, 2])], "loose_dimension": 3},


    ]

    for test in tests:
        print(f"\n{'='*60}")
        print(f"Testing {test['dim']}D: {test['label']}")
        print(f"Start: {test['x_0']}, Goal: {test['x_f']}, Obstacles: {test['obstacles']}")
        print(f"{'='*60}")

        visualizer = RayTracerVisualizer()
        path, front_cells, intersections, y_history, hit, goal_reached = visualizer.traverse(
            test['x_0'], test['x_f'], test['obstacles'], loose_dimension=test.get('loose_dimension', 0)
        )

        print(f"\nTraversal Results for {test['label']}:")
        print(f"  Obstacle hit: {hit}")
        print(f"  Goal reached: {goal_reached}")

        plot_trace(
            test['dim'], test['label'],
            test['x_0'], test['x_f'],
            path, front_cells, intersections, y_history,
            test['obstacles'],
            goal_reached,
            test.get('loose_dimension', 0)
        )