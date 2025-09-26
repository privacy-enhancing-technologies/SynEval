#!/usr/bin/env python3
"""
Test runner script for SynEval framework
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    else:
        print(f"❌ {description} - FAILED")
        if result.stderr:
            print("Error:")
            print(result.stderr)
        if result.stdout:
            print("Output:")
            print(result.stdout)
    
    return result.returncode == 0

def main():
    """Main test runner"""
    print("🧪 SynEval Test Runner")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Test commands
    tests = [
        ("python -c 'import tests.conftest; print(\"conftest.py works\")'", "Test configuration"),
        ("python -m pytest tests/test_syneval.py -v", "Basic functionality tests"),
        ("python -m pytest tests/test_metrics_validation.py -v", "Metrics validation tests"),
        ("python -c 'from fidelity import FidelityEvaluator; print(\"FidelityEvaluator import OK\")'", "Fidelity module import"),
        ("python -c 'from utility import UtilityEvaluator; print(\"UtilityEvaluator import OK\")'", "Utility module import"),
        ("python -c 'from diversity import DiversityEvaluator; print(\"DiversityEvaluator import OK\")'", "Diversity module import"),
    ]
    
    # Run tests
    passed = 0
    total = len(tests)
    
    for cmd, description in tests:
        if run_command(cmd, description):
            passed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
