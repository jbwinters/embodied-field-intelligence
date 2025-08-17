#!/usr/bin/env python3
"""Test script to verify all commands work."""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✓ SUCCESS")
            # Show first few lines of output
            lines = result.stdout.split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"✗ FAILED (exit code: {result.returncode})")
            print(f"Error: {result.stderr[:200]}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def main():
    """Run test suite."""
    tests = [
        ("python cli.py demo --episodes 2 --max-steps 50", 
         "Demo mode with 2 episodes"),
        
        ("python cli.py eval --episodes 5 --seeds 2 --max-steps 50 --out test_eval", 
         "Evaluation mode"),
        
        ("python cli.py suite --episodes 3 --seeds 1 --max-steps 30 --out test_suite",
         "Ablation suite mode"),
        
        ("python cli.py gym-register",
         "Gym environment registration"),
         
        ("python -c 'from efi import *; print(\"Import successful\")'",
         "Package import test"),
         
        ("python -c 'from efi.visualization import InteractiveViewer; print(\"Interactive viewer imported\")'",
         "Interactive viewer import"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    for desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {desc}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())