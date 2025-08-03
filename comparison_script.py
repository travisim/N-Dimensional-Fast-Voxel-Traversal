#!/usr/bin/env python3
"""
Comparison script between original and optimized n-dimensional fast voxel traversal implementations.

This script compares:
1. Correctness - Do both implementations produce the same results?
2. Performance - Which implementation is faster?
3. Memory usage - Which implementation uses less memory?
4. Code complexity - Analysis of the implementations
"""

import time
import numpy as np
import tracemalloc
import sys
import random
from typing import List, Tuple, Dict, Any, Optional
import gc
from contextlib import contextmanager

# Import both implementations
try:
    from n_dimensional_fast_voxel_traversal import RayTracerVisualizer as OriginalRayTracerVisualizer
    from n_dimensional_fast_voxel_traversal import RayTracer as OriginalRayTracer
    from n_dimensional_fast_voxel_traversal import ObstacleRayTracer as OriginalObstacleRayTracer
    print("✓ Successfully imported original implementation")
except ImportError as e:
    print(f"✗ Failed to import original implementation: {e}")
    sys.exit(1)

try:
    from n_dimensional_fast_voxel_traversal_optimized import RayTracerVisualizer as OptimizedRayTracerVisualizer
    from n_dimensional_fast_voxel_traversal_optimized import RayTracer as OptimizedRayTracer
    from n_dimensional_fast_voxel_traversal_optimized import ObstacleRayTracer as OptimizedObstacleRayTracer
    print("✓ Successfully imported optimized implementation")
except ImportError as e:
    print(f"✗ Failed to import optimized implementation: {e}")
    sys.exit(1)


class ComparisonResult:
    """Container for comparison results"""
    def __init__(self):
        self.correctness_tests = []
        self.performance_tests = []
        self.memory_tests = []
        self.errors = []

    def add_correctness_test(self, test_name: str, passed: bool, details: str = ""):
        self.correctness_tests.append({
            'test_name': test_name,
            'passed': passed,
            'details': details
        })

    def add_performance_test(self, test_name: str, original_time: float, optimized_time: float, speedup: float):
        self.performance_tests.append({
            'test_name': test_name,
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': speedup
        })

    def add_memory_test(self, test_name: str, original_memory: float, optimized_memory: float, reduction: float):
        self.memory_tests.append({
            'test_name': test_name,
            'original_memory': original_memory,
            'optimized_memory': optimized_memory,
            'reduction': reduction
        })

    def add_error(self, test_name: str, error: str):
        self.errors.append({
            'test_name': test_name,
            'error': error
        })

    def print_summary(self):
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Correctness Summary
        print("\n📋 CORRECTNESS TESTS:")
        passed_tests = sum(1 for test in self.correctness_tests if test['passed'])
        total_tests = len(self.correctness_tests)
        print(f"   Passed: {passed_tests}/{total_tests}")
        
        for test in self.correctness_tests:
            status = "✓" if test['passed'] else "✗"
            print(f"   {status} {test['test_name']}")
            if test['details'] and not test['passed']:
                print(f"     Details: {test['details']}")
        
        # Performance Summary
        print("\n⚡ PERFORMANCE TESTS:")
        if self.performance_tests:
            avg_speedup = np.mean([test['speedup'] for test in self.performance_tests])
            print(f"   Average speedup: {avg_speedup:.2f}x")
            
            for test in self.performance_tests:
                print(f"   {test['test_name']}:")
                print(f"     Original: {test['original_time']:.4f}s")
                print(f"     Optimized: {test['optimized_time']:.4f}s")
                print(f"     Speedup: {test['speedup']:.2f}x")
        else:
            print("   No performance tests completed")
        
        # Memory Summary
        print("\n🧠 MEMORY TESTS:")
        if self.memory_tests:
            avg_reduction = np.mean([test['reduction'] for test in self.memory_tests])
            print(f"   Average memory reduction: {avg_reduction:.1f}%")
            
            for test in self.memory_tests:
                print(f"   {test['test_name']}:")
                print(f"     Original: {test['original_memory']:.2f} MB")
                print(f"     Optimized: {test['optimized_memory']:.2f} MB")
                print(f"     Reduction: {test['reduction']:.1f}%")
        else:
            print("   No memory tests completed")
        
        # Errors
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"   {error['test_name']}: {error['error']}")


@contextmanager
def measure_memory():
    """Context manager to measure memory usage"""
    tracemalloc.start()
    yield
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024  # Convert to MB


def arrays_equal(arr1, arr2, tolerance=1e-10):
    """Compare arrays with tolerance"""
    if arr1 is None and arr2 is None:
        return True
    if arr1 is None or arr2 is None:
        return False
    
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    if arr1.shape != arr2.shape:
        return False
    
    return np.allclose(arr1, arr2, atol=tolerance)


def lists_of_arrays_equal(list1, list2, tolerance=1e-10):
    """Compare lists of arrays with tolerance"""
    if len(list1) != len(list2):
        return False
    
    for arr1, arr2 in zip(list1, list2):
        if not arrays_equal(arr1, arr2, tolerance):
            return False
    
    return True


class ComparisonTester:
    def __init__(self):
        self.result = ComparisonResult()
        
    def test_basic_ray_tracer_correctness(self):
        """Test basic RayTracer functionality"""
        print("\n🔍 Testing basic RayTracer correctness...")
        
        test_cases = [
            # Simple 2D cases
            (np.array([0.0, 0.0]), np.array([3.0, 3.0])),
            (np.array([1.5, 2.5]), np.array([4.5, 6.5])),
            (np.array([0.0, 0.0]), np.array([0.0, 5.0])),  # Vertical line
            (np.array([0.0, 0.0]), np.array([5.0, 0.0])),  # Horizontal line
            
            # 3D cases
            (np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0])),
            (np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6])),
            
            # Higher dimensions
            (np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0])),
        ]
        
        for i, (x_0, x_f) in enumerate(test_cases):
            try:
                # Test original implementation
                original_tracer = OriginalRayTracer()
                original_tracer.init(x_0, x_f)
                
                # Test optimized implementation
                optimized_tracer = OptimizedRayTracer()
                optimized_tracer.init(x_0, x_f)
                
                # Compare initial states
                states_match = True
                details = []
                
                # Compare basic properties
                if original_tracer.n != optimized_tracer.n:
                    states_match = False
                    details.append(f"Dimension mismatch: {original_tracer.n} vs {optimized_tracer.n}")
                
                # Compare trajectories step by step
                step = 0
                while not original_tracer.reached() and not optimized_tracer.reached() and step < 1000:
                    # Compare current coordinates
                    orig_coords = original_tracer.coords()
                    opt_coords = optimized_tracer.coords()
                    
                    if not arrays_equal(orig_coords, opt_coords):
                        states_match = False
                        details.append(f"Step {step}: coords differ - orig: {orig_coords}, opt: {opt_coords}")
                        break
                    
                    # Compare front cells
                    orig_front = original_tracer.front_cells()
                    opt_front = optimized_tracer.front_cells()
                    
                    if not lists_of_arrays_equal(orig_front, opt_front):
                        states_match = False
                        details.append(f"Step {step}: front cells differ")
                        break
                    
                    # Step both tracers
                    original_tracer.next()
                    optimized_tracer.next()
                    step += 1
                
                # Check if both reached the end
                if original_tracer.reached() != optimized_tracer.reached():
                    states_match = False
                    details.append(f"End state mismatch: orig reached: {original_tracer.reached()}, opt reached: {optimized_tracer.reached()}")
                
                self.result.add_correctness_test(
                    f"Basic RayTracer Test {i+1} ({len(x_0)}D)",
                    states_match,
                    "; ".join(details) if details else ""
                )
                
            except Exception as e:
                self.result.add_error(f"Basic RayTracer Test {i+1}", str(e))

    def test_obstacle_ray_tracer_correctness(self):
        """Test ObstacleRayTracer functionality"""
        print("\n🚧 Testing ObstacleRayTracer correctness...")
        
        test_cases = [
            # Simple 2D case with obstacles
            {
                'x_0': np.array([0.0, 0.0]),
                'x_f': np.array([5.0, 5.0]),
                'obstacles': [np.array([2, 2]), np.array([3, 3])],
                'loose_dimension': 1
            },
            # 3D case
            {
                'x_0': np.array([0.0, 0.0, 0.0]),
                'x_f': np.array([3.0, 3.0, 3.0]),
                'obstacles': [np.array([1, 1, 1]), np.array([2, 2, 2])],
                'loose_dimension': 1
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                # Test original implementation
                original_visualizer = OriginalRayTracerVisualizer()
                orig_result = original_visualizer.traverse(
                    test_case['x_0'], 
                    test_case['x_f'], 
                    test_case['obstacles'], 
                    test_case['loose_dimension']
                )
                
                # Test optimized implementation
                optimized_visualizer = OptimizedRayTracerVisualizer()
                opt_result = optimized_visualizer.traverse(
                    test_case['x_0'], 
                    test_case['x_f'], 
                    test_case['obstacles'], 
                    test_case['loose_dimension']
                )
                
                # Compare results
                results_match = True
                details = []
                
                # Compare paths
                orig_path, orig_front_cells, orig_intersections, orig_y_coords, orig_obstacle_hit, orig_goal_reached = orig_result
                opt_path, opt_front_cells, opt_intersections, opt_y_coords, opt_obstacle_hit, opt_goal_reached = opt_result
                
                if not lists_of_arrays_equal(orig_path, opt_path):
                    results_match = False
                    details.append("Paths differ")
                
                if orig_obstacle_hit != opt_obstacle_hit:
                    results_match = False
                    details.append(f"Obstacle hit status differs: {orig_obstacle_hit} vs {opt_obstacle_hit}")
                
                if orig_goal_reached != opt_goal_reached:
                    results_match = False
                    details.append(f"Goal reached status differs: {orig_goal_reached} vs {opt_goal_reached}")
                
                self.result.add_correctness_test(
                    f"ObstacleRayTracer Test {i+1} ({len(test_case['x_0'])}D)",
                    results_match,
                    "; ".join(details) if details else ""
                )
                
            except Exception as e:
                self.result.add_error(f"ObstacleRayTracer Test {i+1}", str(e))

    def test_performance(self):
        """Test performance comparison"""
        print("\n⚡ Testing performance...")
        
        test_cases = [
            # 2D performance test
            {
                'name': '2D Long Distance',
                'x_0': np.array([0.0, 0.0]),
                'x_f': np.array([100.0, 100.0]),
                'iterations': 100
            },
            # 3D performance test
            {
                'name': '3D Medium Distance',
                'x_0': np.array([0.0, 0.0, 0.0]),
                'x_f': np.array([50.0, 50.0, 50.0]),
                'iterations': 50
            },
            # Higher dimension test
            {
                'name': '4D Short Distance',
                'x_0': np.array([0.0, 0.0, 0.0, 0.0]),
                'x_f': np.array([10.0, 10.0, 10.0, 10.0]),
                'iterations': 20
            }
        ]
        
        for test_case in test_cases:
            try:
                # Test original implementation
                start_time = time.time()
                for _ in range(test_case['iterations']):
                    tracer = OriginalRayTracer()
                    tracer.init(test_case['x_0'], test_case['x_f'])
                    while not tracer.reached():
                        tracer.next()
                original_time = time.time() - start_time
                
                # Test optimized implementation
                start_time = time.time()
                for _ in range(test_case['iterations']):
                    tracer = OptimizedRayTracer()
                    tracer.init(test_case['x_0'], test_case['x_f'])
                    while not tracer.reached():
                        tracer.next()
                optimized_time = time.time() - start_time
                
                speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
                
                self.result.add_performance_test(
                    test_case['name'],
                    original_time,
                    optimized_time,
                    speedup
                )
                
            except Exception as e:
                self.result.add_error(f"Performance Test {test_case['name']}", str(e))

    def test_memory_usage(self):
        """Test memory usage comparison"""
        print("\n🧠 Testing memory usage...")
        
        test_cases = [
            {
                'name': '2D Memory Test',
                'x_0': np.array([0.0, 0.0]),
                'x_f': np.array([50.0, 50.0])
            },
            {
                'name': '3D Memory Test',
                'x_0': np.array([0.0, 0.0, 0.0]),
                'x_f': np.array([30.0, 30.0, 30.0])
            }
        ]
        
        for test_case in test_cases:
            try:
                # Test original implementation memory usage
                gc.collect()
                tracemalloc.start()
                
                tracer = OriginalRayTracer()
                tracer.init(test_case['x_0'], test_case['x_f'])
                while not tracer.reached():
                    tracer.next()
                
                current, peak_original = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_original_mb = peak_original / 1024 / 1024
                
                # Test optimized implementation memory usage
                gc.collect()
                tracemalloc.start()
                
                tracer = OptimizedRayTracer()
                tracer.init(test_case['x_0'], test_case['x_f'])
                while not tracer.reached():
                    tracer.next()
                
                current, peak_optimized = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_optimized_mb = peak_optimized / 1024 / 1024
                
                reduction = ((peak_original_mb - peak_optimized_mb) / peak_original_mb) * 100
                
                self.result.add_memory_test(
                    test_case['name'],
                    peak_original_mb,
                    peak_optimized_mb,
                    reduction
                )
                
            except Exception as e:
                self.result.add_error(f"Memory Test {test_case['name']}", str(e))

    def test_edge_cases(self):
        """Test edge cases"""
        print("\n🔬 Testing edge cases...")
        
        edge_cases = [
            # Zero distance
            (np.array([1.0, 1.0]), np.array([1.0, 1.0])),
            # Single step
            (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            # Negative coordinates
            (np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
            # Mixed positive/negative
            (np.array([2.0, -2.0]), np.array([-2.0, 2.0])),
        ]
        
        for i, (x_0, x_f) in enumerate(edge_cases):
            try:
                # Test both implementations
                original_tracer = OriginalRayTracer()
                original_tracer.init(x_0, x_f)
                
                optimized_tracer = OptimizedRayTracer()
                optimized_tracer.init(x_0, x_f)
                
                # Simple check - both should initialize without error
                self.result.add_correctness_test(
                    f"Edge Case {i+1}: {x_0} -> {x_f}",
                    True,
                    "Both implementations handled edge case"
                )
                
            except Exception as e:
                self.result.add_error(f"Edge Case {i+1}", str(e))

    def run_all_tests(self):
        """Run all comparison tests"""
        print("Starting comprehensive comparison between implementations...")
        
        self.test_basic_ray_tracer_correctness()
        self.test_obstacle_ray_tracer_correctness()
        self.test_performance()
        self.test_memory_usage()
        self.test_edge_cases()
        
        self.result.print_summary()


def main():
    """Main function to run the comparison"""
    print("N-Dimensional Fast Voxel Traversal Implementation Comparison")
    print("=" * 80)
    
    tester = ComparisonTester()
    tester.run_all_tests()
    
    print("\n" + "="*80)
    print("Comparison completed!")
    print("="*80)


if __name__ == "__main__":
    main()
