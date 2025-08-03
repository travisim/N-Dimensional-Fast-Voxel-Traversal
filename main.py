import numpy as np
from n_dimensional_fast_voxel_traversal import RayTracerVisualizer
from plotting import plot_trace
import csv
import json

if __name__ == "__main__":
    # ============================================================================
    # METHODICAL BENCHMARK TEST SUITE FOR N-DIMENSIONAL FAST VOXEL TRAVERSAL
    # ============================================================================
    # 
    # This test suite is designed as a comprehensive benchmark with methodical 
    # coverage of all essential scenarios without redundancy. Each test case serves 
    # a specific purpose in validating the algorithm's correctness and performance.
    #
    # ORGANIZATION BY COORDINATE TYPES:
    # 1. INTEGER COORDINATES (2D first, then 3D)
    # 2. FLOAT COORDINATES
    #    2.1 Float .5 Coordinates (2D first, then 3D)
    #    2.2 Float Other Coordinates (2D first, then 3D)
    # 3. OBSTACLE INTERACTION TESTS
    # 4. EDGE CASES AND BOUNDARY CONDITIONS  
    # 5. LOOSE DIMENSION VALIDATION
    # 6. PERFORMANCE AND STRESS TESTS
    #
    # COORDINATE-BASED ORGANIZATION:
    # - Total directional coverage: 4×2 + 8×2 = 24 comprehensive directional test cases
    # - Organized by coordinate type first, then dimensionality (2D/3D)
    # ============================================================================
    
    tests = [
        # ============================================================================
        # SECTION 1: INTEGER COORDINATES
        # Purpose: Validate core ray traversal with integer coordinates
        # Organization: 2D tests first, then 3D tests
        # ============================================================================
        
        # 1.1 Integer 2D Tests
        {"dim": 2, "label": "Basic-2D-Int-Diagonal", "x_0": np.array([1, 1]), "x_f": np.array([5, 5]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Basic-2D-Int-Horizontal", "x_0": np.array([1, 2]), "x_f": np.array([5, 2]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Basic-2D-Int-Vertical", "x_0": np.array([2, 1]), "x_f": np.array([2, 5]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Basic-2D-Int-AntiDiagonal", "x_0": np.array([1, 5]), "x_f": np.array([5, 1]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Direction-2D-Int-East-0deg", "x_0": np.array([3, 3]), "x_f": np.array([7, 3]), "obstacles": [], "loose_dimension": 1},  # East (0°)
        {"dim": 2, "label": "Direction-2D-Int-North-90deg", "x_0": np.array([3, 3]), "x_f": np.array([3, 7]), "obstacles": [], "loose_dimension": 1},  # North (90°)
        {"dim": 2, "label": "Direction-2D-Int-West-180deg", "x_0": np.array([7, 3]), "x_f": np.array([3, 3]), "obstacles": [], "loose_dimension": 1},  # West (180°)
        {"dim": 2, "label": "Direction-2D-Int-South-270deg", "x_0": np.array([3, 7]), "x_f": np.array([3, 3]), "obstacles": [], "loose_dimension": 1},  # South (270°)
        
        # 1.2 Integer 3D Tests
        {"dim": 3, "label": "Basic-3D-Int-Diagonal", "x_0": np.array([1, 1, 1]), "x_f": np.array([5, 5, 5]), "obstacles": [], "loose_dimension": 1},
        {"dim": 3, "label": "Basic-3D-Int-AxisX", "x_0": np.array([1, 2, 2]), "x_f": np.array([5, 2, 2]), "obstacles": [], "loose_dimension": 1},
        {"dim": 3, "label": "Basic-3D-Int-AxisY", "x_0": np.array([2, 1, 2]), "x_f": np.array([2, 5, 2]), "obstacles": [], "loose_dimension": 1},
        {"dim": 3, "label": "Basic-3D-Int-AxisZ", "x_0": np.array([2, 2, 1]), "x_f": np.array([2, 2, 5]), "obstacles": [], "loose_dimension": 1},
        {"dim": 3, "label": "Basic-3D-Int-PlaneDiagXY", "x_0": np.array([1, 1, 2]), "x_f": np.array([5, 5, 2]), "obstacles": [], "loose_dimension": 1},
        {"dim": 3, "label": "Direction-3D-Int-PosX", "x_0": np.array([3, 3, 3]), "x_f": np.array([7, 3, 3]), "obstacles": [], "loose_dimension": 1},  # +X direction
        {"dim": 3, "label": "Direction-3D-Int-NegX", "x_0": np.array([7, 3, 3]), "x_f": np.array([3, 3, 3]), "obstacles": [], "loose_dimension": 1},  # -X direction
        {"dim": 3, "label": "Direction-3D-Int-PosY", "x_0": np.array([3, 3, 3]), "x_f": np.array([3, 7, 3]), "obstacles": [], "loose_dimension": 1},  # +Y direction
        {"dim": 3, "label": "Direction-3D-Int-NegY", "x_0": np.array([3, 7, 3]), "x_f": np.array([3, 3, 3]), "obstacles": [], "loose_dimension": 1},  # -Y direction
        {"dim": 3, "label": "Direction-3D-Int-PosZ", "x_0": np.array([3, 3, 3]), "x_f": np.array([3, 3, 7]), "obstacles": [], "loose_dimension": 1},  # +Z direction
        {"dim": 3, "label": "Direction-3D-Int-NegZ", "x_0": np.array([3, 3, 7]), "x_f": np.array([3, 3, 3]), "obstacles": [], "loose_dimension": 1},  # -Z direction
        {"dim": 3, "label": "Direction-3D-Int-MainDiagonal", "x_0": np.array([3, 3, 3]), "x_f": np.array([7, 7, 7]), "obstacles": [], "loose_dimension": 1},  # Main diagonal
        {"dim": 3, "label": "Direction-3D-Int-AntiDiagonal", "x_0": np.array([7, 7, 7]), "x_f": np.array([3, 3, 3]), "obstacles": [], "loose_dimension": 1},  # Anti-diagonal
        
        # ============================================================================
        # SECTION 2: FLOAT COORDINATES
        # Purpose: Validate core ray traversal with float coordinates
        # Organization: .5 coordinates first, then other float values; 2D first, then 3D
        # ============================================================================
        
        # 2.1 Float .5 Coordinates - 2D Tests
        {"dim": 2, "label": "Basic-2D-Float-Diagonal", "x_0": np.array([1.5, 1.5]), "x_f": np.array([5.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Basic-2D-Float-Horizontal", "x_0": np.array([1.5, 2.5]), "x_f": np.array([5.5, 2.5]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Direction-2D-Float-East-0deg", "x_0": np.array([3.5, 3.5]), "x_f": np.array([7.5, 3.5]), "obstacles": [], "loose_dimension": 1},  # East (0°)
        {"dim": 2, "label": "Direction-2D-Float-North-90deg", "x_0": np.array([3.5, 3.5]), "x_f": np.array([3.5, 7.5]), "obstacles": [], "loose_dimension": 1},  # North (90°)
        {"dim": 2, "label": "Direction-2D-Float-West-180deg", "x_0": np.array([7.5, 3.5]), "x_f": np.array([3.5, 3.5]), "obstacles": [], "loose_dimension": 1},  # West (180°)
        {"dim": 2, "label": "Direction-2D-Float-South-270deg", "x_0": np.array([3.5, 7.5]), "x_f": np.array([3.5, 3.5]), "obstacles": [], "loose_dimension": 1},  # South (270°)
        
        # 2.2 Float .5 Coordinates - 3D Tests
        {"dim": 3, "label": "Basic-3D-Float-Diagonal", "x_0": np.array([1.5, 1.5, 1.5]), "x_f": np.array([5.5, 5.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        {"dim": 3, "label": "Basic-3D-Float-AxisZ", "x_0": np.array([2.5, 2.5, 1.5]), "x_f": np.array([2.5, 2.5, 5.5]), "obstacles": [], "loose_dimension": 1},
        {"dim": 3, "label": "Direction-3D-Float-PosX", "x_0": np.array([3.5, 3.5, 3.5]), "x_f": np.array([7.5, 3.5, 3.5]), "obstacles": [], "loose_dimension": 1},  # +X direction
        {"dim": 3, "label": "Direction-3D-Float-NegX", "x_0": np.array([7.5, 3.5, 3.5]), "x_f": np.array([3.5, 3.5, 3.5]), "obstacles": [], "loose_dimension": 1},  # -X direction
        {"dim": 3, "label": "Direction-3D-Float-PosY", "x_0": np.array([3.5, 3.5, 3.5]), "x_f": np.array([3.5, 7.5, 3.5]), "obstacles": [], "loose_dimension": 1},  # +Y direction
        {"dim": 3, "label": "Direction-3D-Float-NegY", "x_0": np.array([3.5, 7.5, 3.5]), "x_f": np.array([3.5, 3.5, 3.5]), "obstacles": [], "loose_dimension": 1},  # -Y direction
        {"dim": 3, "label": "Direction-3D-Float-PosZ", "x_0": np.array([3.5, 3.5, 3.5]), "x_f": np.array([3.5, 3.5, 7.5]), "obstacles": [], "loose_dimension": 1},  # +Z direction
        {"dim": 3, "label": "Direction-3D-Float-NegZ", "x_0": np.array([3.5, 3.5, 7.5]), "x_f": np.array([3.5, 3.5, 3.5]), "obstacles": [], "loose_dimension": 1},  # -Z direction
        {"dim": 3, "label": "Direction-3D-Float-MainDiagonal", "x_0": np.array([3.5, 3.5, 3.5]), "x_f": np.array([7.5, 7.5, 7.5]), "obstacles": [], "loose_dimension": 1},  # Main diagonal
        {"dim": 3, "label": "Direction-3D-Float-AntiDiagonal", "x_0": np.array([7.5, 7.5, 7.5]), "x_f": np.array([3.5, 3.5, 3.5]), "obstacles": [], "loose_dimension": 1},  # Anti-diagonal
        
        # 2.3 Float Other Coordinates - 2D Tests
        {"dim": 2, "label": "Basic-2D-Float-AsymmetricRatio", "x_0": np.array([1.2, 1.8]), "x_f": np.array([5.7, 6.3]), "obstacles": [], "loose_dimension": 1},
        # Additional 2D Directional Tests with 3 decimal places (non-.5 coordinates)
        {"dim": 2, "label": "Direction-2D-Float3dp-East-0deg", "x_0": np.array([3.125, 3.125]), "x_f": np.array([7.125, 3.125]), "obstacles": [], "loose_dimension": 1},  # East (0°)
        {"dim": 2, "label": "Direction-2D-Float3dp-North-90deg", "x_0": np.array([3.375, 3.375]), "x_f": np.array([3.375, 7.375]), "obstacles": [], "loose_dimension": 1},  # North (90°)
        {"dim": 2, "label": "Direction-2D-Float3dp-West-180deg", "x_0": np.array([7.750, 3.250]), "x_f": np.array([3.750, 3.250]), "obstacles": [], "loose_dimension": 1},  # West (180°)
        {"dim": 2, "label": "Direction-2D-Float3dp-South-270deg", "x_0": np.array([3.875, 7.625]), "x_f": np.array([3.875, 3.625]), "obstacles": [], "loose_dimension": 1},  # South (270°)
        
        # 2.4 Float Other Coordinates - 3D Tests
        {"dim": 3, "label": "Basic-3D-Float-AsymmetricRatio", "x_0": np.array([1.2, 1.8, 1.1]), "x_f": np.array([5.7, 6.3, 5.9]), "obstacles": [], "loose_dimension": 1},
        # Additional 3D Directional Tests with 3 decimal places (non-.5 coordinates)
        {"dim": 3, "label": "Direction-3D-Float3dp-PosX", "x_0": np.array([3.125, 3.125, 3.125]), "x_f": np.array([7.125, 3.125, 3.125]), "obstacles": [], "loose_dimension": 1},  # +X direction
        {"dim": 3, "label": "Direction-3D-Float3dp-NegX", "x_0": np.array([7.375, 3.375, 3.375]), "x_f": np.array([3.375, 3.375, 3.375]), "obstacles": [], "loose_dimension": 1},  # -X direction
        {"dim": 3, "label": "Direction-3D-Float3dp-PosY", "x_0": np.array([3.250, 3.250, 3.250]), "x_f": np.array([3.250, 7.250, 3.250]), "obstacles": [], "loose_dimension": 1},  # +Y direction
        {"dim": 3, "label": "Direction-3D-Float3dp-NegY", "x_0": np.array([3.625, 7.625, 3.625]), "x_f": np.array([3.625, 3.625, 3.625]), "obstacles": [], "loose_dimension": 1},  # -Y direction
        {"dim": 3, "label": "Direction-3D-Float3dp-PosZ", "x_0": np.array([3.750, 3.750, 3.750]), "x_f": np.array([3.750, 3.750, 7.750]), "obstacles": [], "loose_dimension": 1},  # +Z direction
        {"dim": 3, "label": "Direction-3D-Float3dp-NegZ", "x_0": np.array([3.875, 3.875, 7.875]), "x_f": np.array([3.875, 3.875, 3.875]), "obstacles": [], "loose_dimension": 1},  # -Z direction
        {"dim": 3, "label": "Direction-3D-Float3dp-MainDiagonal", "x_0": np.array([3.125, 3.125, 3.125]), "x_f": np.array([7.125, 7.125, 7.125]), "obstacles": [], "loose_dimension": 1},  # Main diagonal
        {"dim": 3, "label": "Direction-3D-Float3dp-AntiDiagonal", "x_0": np.array([7.875, 7.875, 7.875]), "x_f": np.array([3.875, 3.875, 3.875]), "obstacles": [], "loose_dimension": 1},  # Anti-diagonal
        
        # ============================================================================
        # SECTION 3: OBSTACLE INTERACTION TESTS
        # Purpose: Validate obstacle detection and navigation behavior
        # Organization: Integer coordinates first, then float coordinates; 2D first, then 3D
        # ============================================================================
        
        # 3.1 Integer Coordinates with Obstacles - 2D Tests
        {"dim": 2, "label": "Obstacle-2D-PathBlocked", "x_0": np.array([1, 1]), "x_f": np.array([5, 5]), "obstacles": [np.array([3, 3])], "loose_dimension": 1},
        {"dim": 2, "label": "Obstacle-2D-StartBlocked", "x_0": np.array([1, 1]), "x_f": np.array([5, 5]), "obstacles": [np.array([1, 1])], "loose_dimension": 1},
        {"dim": 2, "label": "Obstacle-2D-GoalBlocked", "x_0": np.array([1, 1]), "x_f": np.array([5, 5]), "obstacles": [np.array([5, 5])], "loose_dimension": 1},
        {"dim": 2, "label": "Obstacle-2D-MultipleBlocks", "x_0": np.array([1, 2]), "x_f": np.array([5, 2]), "obstacles": [np.array([3, 2]), np.array([4, 2])], "loose_dimension": 1},
        
        # 3.2 Integer Coordinates with Obstacles - 3D Tests
        {"dim": 3, "label": "Obstacle-3D-PathBlocked", "x_0": np.array([1, 1, 1]), "x_f": np.array([5, 5, 5]), "obstacles": [np.array([3, 3, 3])], "loose_dimension": 1},
        {"dim": 3, "label": "Obstacle-3D-ComplexScenario", "x_0": np.array([1, 1, 1]), "x_f": np.array([3, 3, 3]), "obstacles": [np.array([2, 1, 1]), np.array([1, 1, 2]), np.array([1, 2, 1])], "loose_dimension": 1},
        
        # 3.3 Float Coordinates with Obstacles - 2D Tests
        {"dim": 2, "label": "Obstacle-2D-Float-PathBlocked", "x_0": np.array([1.5, 1.5]), "x_f": np.array([5.5, 5.5]), "obstacles": [np.array([3.0, 3.0])], "loose_dimension": 1},
        
        # 3.4 Float Coordinates with Obstacles - 3D Tests
        {"dim": 3, "label": "Obstacle-3D-Float-PathBlocked", "x_0": np.array([1.5, 1.5, 1.5]), "x_f": np.array([5.5, 5.5, 5.5]), "obstacles": [np.array([3.0, 3.0, 3.0])], "loose_dimension": 1},
        
        # ============================================================================
        # SECTION 4: EDGE CASES AND BOUNDARY CONDITIONS
        # Purpose: Test algorithm robustness at boundary conditions
        # Organization: Integer coordinates first, then float coordinates; 2D first, then 3D
        # ============================================================================
        
        # 4.1 Integer Coordinates Edge Cases - 2D Tests
        {"dim": 2, "label": "Edge-2D-ZeroMovement", "x_0": np.array([2, 2]), "x_f": np.array([2, 2]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Edge-2D-ZeroMovementAtObstacle", "x_0": np.array([2, 2]), "x_f": np.array([2, 2]), "obstacles": [np.array([2, 2])], "loose_dimension": 1},
        {"dim": 2, "label": "Edge-2D-SingleStep", "x_0": np.array([2, 2]), "x_f": np.array([3, 3]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Edge-2D-NegativeCoords", "x_0": np.array([-2, -2]), "x_f": np.array([2, 2]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Edge-2D-LargeCoords", "x_0": np.array([100, 100]), "x_f": np.array([105, 105]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Edge-2D-SteepAngle", "x_0": np.array([1, 1]), "x_f": np.array([2, 10]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Edge-2D-ObstacleBoundaryStart", "x_0": np.array([1.0, 1.0]), "x_f": np.array([4.0, 4.0]), "obstacles": [np.array([1, 1])], "loose_dimension": 1},
        {"dim": 2, "label": "Edge-2D-ObstacleBoundaryEnd", "x_0": np.array([1.0, 1.0]), "x_f": np.array([4.0, 4.0]), "obstacles": [np.array([4, 4])], "loose_dimension": 1},
        
        # 4.2 Integer Coordinates Edge Cases - 3D Tests
        {"dim": 3, "label": "Edge-3D-ZeroMovement", "x_0": np.array([2, 2, 2]), "x_f": np.array([2, 2, 2]), "obstacles": [], "loose_dimension": 1},
        {"dim": 3, "label": "Edge-3D-SingleStepZ", "x_0": np.array([2, 2, 2]), "x_f": np.array([2, 2, 3]), "obstacles": [], "loose_dimension": 1},
        
        # 4.3 Float Coordinates Edge Cases - 2D Tests  
        {"dim": 2, "label": "Edge-2D-IntToFloat", "x_0": np.array([1, 1]), "x_f": np.array([4.5, 4.5]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Edge-2D-FloatToInt", "x_0": np.array([1.5, 1.5]), "x_f": np.array([4, 4]), "obstacles": [], "loose_dimension": 1},
        {"dim": 2, "label": "Edge-2D-FloatPrecision", "x_0": np.array([1.0000001, 1.0000001]), "x_f": np.array([2.9999999, 2.9999999]), "obstacles": [], "loose_dimension": 1},
        
        # 4.4 Float Coordinates Edge Cases - 3D Tests
        {"dim": 3, "label": "Edge-3D-IntToFloat", "x_0": np.array([1, 1, 1]), "x_f": np.array([4.5, 4.5, 4.5]), "obstacles": [], "loose_dimension": 1},
        
        # ============================================================================
        # SECTION 5: LOOSE DIMENSION VALIDATION
        # Purpose: Test loose dimension parameter functionality
        # Organization: Integer coordinates first, then 2D first, then 3D
        # ============================================================================
        
        # 5.1 Integer Coordinates Loose Dimension Tests
        {"dim": 2, "label": "Loose-2D-Dim2", "x_0": np.array([1, 1]), "x_f": np.array([4, 4]), "obstacles": [np.array([2, 2]), np.array([3, 3])], "loose_dimension": 2},
        {"dim": 3, "label": "Loose-3D-Dim1", "x_0": np.array([1, 1, 1]), "x_f": np.array([3, 3, 3]), "obstacles": [np.array([2, 1, 1]), np.array([1, 1, 2]), np.array([1, 2, 1])], "loose_dimension": 1},
        {"dim": 3, "label": "Loose-3D-Dim2", "x_0": np.array([1, 1, 1]), "x_f": np.array([3, 3, 3]), "obstacles": [np.array([2, 1, 1]), np.array([1, 1, 2]), np.array([1, 2, 1]), np.array([2, 1, 2]), np.array([1, 2, 2])], "loose_dimension": 2},
        {"dim": 3, "label": "Loose-3D-Dim3", "x_0": np.array([1, 1, 1]), "x_f": np.array([3, 3, 3]), "obstacles": [np.array([2, 1, 1]), np.array([1, 1, 2]), np.array([1, 2, 1]), np.array([2, 1, 2]), np.array([1, 2, 2]), np.array([2, 2, 1])], "loose_dimension": 3},
        
        # ============================================================================
        # SECTION 6: PERFORMANCE AND STRESS TESTS
        # Purpose: Test algorithm performance under challenging conditions
        # Organization: Integer coordinates first, then float coordinates; 2D first, then 3D
        # ============================================================================
        
        # 6.1 Integer Coordinates Performance Tests - 2D
        {"dim": 2, "label": "Perf-2D-LongDistance", "x_0": np.array([0, 0]), "x_f": np.array([20, 20]), "obstacles": [np.array([10, 10])], "loose_dimension": 1},
        {"dim": 2, "label": "Perf-2D-DenseObstacles", "x_0": np.array([0, 0]), "x_f": np.array([5, 5]), "obstacles": [np.array([1, 1]), np.array([1, 2]), np.array([2, 1]), np.array([2, 3]), np.array([3, 2]), np.array([4, 3])], "loose_dimension": 1},
        {"dim": 2, "label": "Perf-2D-AsymmetricRatio", "x_0": np.array([1, 1]), "x_f": np.array([10, 3]), "obstacles": [], "loose_dimension": 1},
        
        # 6.2 Integer Coordinates Performance Tests - 3D
        {"dim": 3, "label": "Perf-3D-LongDistance", "x_0": np.array([0, 0, 0]), "x_f": np.array([15, 15, 15]), "obstacles": [np.array([7, 7, 7])], "loose_dimension": 1},
        {"dim": 3, "label": "Perf-3D-DenseObstacles", "x_0": np.array([0, 0, 0]), "x_f": np.array([4, 4, 4]), "obstacles": [np.array([1, 1, 1]), np.array([1, 2, 1]), np.array([2, 1, 2]), np.array([2, 2, 2]), np.array([3, 1, 3])], "loose_dimension": 2},
        {"dim": 3, "label": "Perf-3D-AsymmetricRatio", "x_0": np.array([1, 1, 1]), "x_f": np.array([2, 4, 8]), "obstacles": [], "loose_dimension": 1},
        
        # 6.3 Float Coordinates Performance Tests - 2D
        {"dim": 2, "label": "Perf-2D-BoundaryNavigation", "x_0": np.array([0.5, 0.5]), "x_f": np.array([4.5, 4.5]), "obstacles": [np.array([0, 0]), np.array([1, 0]), np.array([4, 4]), np.array([4, 5])], "loose_dimension": 1},
        
    ]
    
    all_results = []

    for test in tests:
        print(f"\n{'='*60}")
        print(f"Testing {test['dim']}D: {test['label']}")
        print(f"Start: {test['x_0']}, Goal: {test['x_f']}, Obstacles: {test['obstacles']}")
        print(f"{'='*60}")

        obstacle_set = {tuple(obs) for obs in test['obstacles']}
        is_obstacle_func = lambda coord: tuple(coord) in obstacle_set

        visualizer = RayTracerVisualizer()
        path, front_cells, intersections, y_history, hit, goal_reached = visualizer.traverse(
            test['x_0'], test['x_f'], obstacles=test['obstacles'], loose_dimension=test.get('loose_dimension', 0)
        )

        print(f"\nTraversal Results for {test['label']}:")
        print(f"  Obstacle hit: {hit}")
        print(f"  Goal reached: {goal_reached}")

        # Store results
        result_data = {
            "label": test['label'],
            "dim": test['dim'],
            "x_0": test['x_0'].tolist(),
            "x_f": test['x_f'].tolist(),
            "obstacles": [obs.tolist() for obs in test['obstacles']],
            "path": [p.tolist() for p in path],
            "front_cells": [[fc.tolist() for fc in fcs] for fcs in front_cells],
            "intersections": [i.tolist() for i in intersections],
            "y_history": [y.tolist() for y in y_history],
            "hit": hit,
            "goal_reached": goal_reached
        }
        all_results.append(result_data)

        plot_trace(
            test['dim'], test['label'],
            test['x_0'], test['x_f'],
            path, front_cells, intersections, y_history,
            test['obstacles'],
            goal_reached,
            test.get('loose_dimension', 0)
        )

    # Write to CSV
    output_filename = 'ray_trace_output.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in all_results:
            # Serialize list/array data into JSON strings
            for key, value in result.items():
                if isinstance(value, list):
                    result[key] = json.dumps(value)
            writer.writerow(result)
    
    print(f"\nAll test results saved to {output_filename}")