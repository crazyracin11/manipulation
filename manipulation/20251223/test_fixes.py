#!/usr/bin/env python3
"""
Test script to validate the fixes for re-orient.py
This script tests the JSON serialization fix and basic functionality
"""

import json
import numpy as np
import os
import sys

# Import the JSON serialization function directly
import importlib.util
spec = importlib.util.spec_from_file_location("re_orient", "/root/HW/RL_qizhi-main/manipulation/re-orient.py")
re_orient_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(re_orient_module)
json_serialize = re_orient_module.json_serialize

def test_json_serialization():
    """Test the JSON serialization function with various data types"""
    print("Testing JSON serialization function...")

    test_cases = [
        {"name": "Basic types", "data": {"int": 42, "float": 3.14, "str": "hello", "bool": True}},
        {"name": "Numpy array", "data": np.array([1, 2, 3, 4, 5])},
        {"name": "Nested dict with arrays", "data": {"a": np.array([1, 2]), "b": {"c": np.array([3, 4])}}},
        {"name": "List of arrays", "data": [np.array([1, 2]), np.array([3, 4])]},
        {"name": "Complex structure", "data": {
            "metrics": {
                "episode_reward": np.array(1.23),
                "episode_reward_std": np.array(0.45),
                "nested": {
                    "more_data": np.array([1, 2, 3])
                }
            }
        }}
    ]

    all_passed = True

    for test_case in test_cases:
        try:
            # Test serialization
            serialized = json_serialize(test_case["data"])

            # Test JSON dump
            json_str = json.dumps(serialized)

            # Test JSON load
            loaded = json.loads(json_str)

            print(f"✓ {test_case['name']}: Passed")

        except Exception as e:
            print(f"✗ {test_case['name']}: Failed - {e}")
            all_passed = False

    return all_passed

def test_progress_logging():
    """Test the progress logging functionality"""
    print("\nTesting progress logging...")

    try:
        # Simulate metrics that would come from training
        mock_metrics = {
            "eval/episode_reward": np.array(1.23),
            "eval/episode_reward_std": np.array(0.45),
            "other_metric": np.array([1, 2, 3])
        }

        # Test our serialization approach
        progress_data = {
            'timestamp': "2025-12-19T10:00:00",
            'num_steps': 100,
            'episode_reward': json_serialize(mock_metrics.get("eval/episode_reward", 0)),
            'episode_reward_std': json_serialize(mock_metrics.get("eval/episode_reward_std", 0)),
        }

        serialized_data = json_serialize(progress_data)

        # Try to write to a test file
        test_filename = "test_progress.json"
        with open(test_filename, 'w') as f:
            f.write(json.dumps(serialized_data) + '\n')

        # Clean up
        if os.path.exists(test_filename):
            os.remove(test_filename)

        print("✓ Progress logging test: Passed")
        return True

    except Exception as e:
        print(f"✗ Progress logging test: Failed - {e}")
        return False

def main():
    print("=" * 60)
    print("Testing fixes for re-orient.py")
    print("=" * 60)

    # Run tests
    test1_passed = test_json_serialization()
    test2_passed = test_progress_logging()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✓ All tests passed! The fixes should resolve the JSON serialization issues.")
        print("You can now run: python re-orient.py")
    else:
        print("✗ Some tests failed. Please review the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()