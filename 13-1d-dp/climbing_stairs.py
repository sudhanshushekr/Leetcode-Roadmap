"""
Climbing Stairs - LeetCode Problem 70
https://leetcode.com/problems/climbing-stairs/

You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Example 1:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Example 2:
Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step

Constraints:
- 1 <= n <= 45
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Climbing Stairs
    """
    
    def climbStairs(self, n: int) -> int:
        """
        Main method - returns the most efficient solution
        """
        return self.climb_stairs_optimized_dp(n)
    
    def climb_stairs_recursive(self, n: int) -> int:
        """
        Approach 1: Recursive - Simple but inefficient
        
        Algorithm:
        1. Base cases: n = 1 (1 way), n = 2 (2 ways)
        2. For n > 2: ways(n) = ways(n-1) + ways(n-2)
        3. This is the Fibonacci sequence
        
        Time Complexity: O(2^n) - Exponential due to repeated calculations
        Space Complexity: O(n) - Recursion stack depth
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Very inefficient, exponential time complexity
        """
        if n <= 2:
            return n
        
        return self.climb_stairs_recursive(n - 1) + self.climb_stairs_recursive(n - 2)
    
    def climb_stairs_memoization(self, n: int) -> int:
        """
        Approach 2: Recursive with Memoization - Top-down DP
        
        Algorithm:
        1. Use memoization to cache results
        2. Avoid recalculating the same subproblems
        3. Still recursive but much more efficient
        
        Time Complexity: O(n) - Each subproblem calculated once
        Space Complexity: O(n) - Memoization cache + recursion stack
        
        Analysis:
        - Pros: Avoids repeated calculations, easy to understand
        - Cons: Still uses recursion stack space
        """
        memo = {}
        
        def climb_helper(n):
            if n in memo:
                return memo[n]
            
            if n <= 2:
                return n
            
            memo[n] = climb_helper(n - 1) + climb_helper(n - 2)
            return memo[n]
        
        return climb_helper(n)
    
    def climb_stairs_dp(self, n: int) -> int:
        """
        Approach 3: Dynamic Programming - Bottom-up with array
        
        Algorithm:
        1. Use a DP array to store results
        2. Fill the array iteratively from base cases
        3. dp[i] = dp[i-1] + dp[i-2]
        
        Time Complexity: O(n) - Single pass through array
        Space Complexity: O(n) - DP array
        
        Analysis:
        - Pros: No recursion, clear iterative approach
        - Cons: Uses O(n) space for array
        """
        if n <= 2:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
    
    def climb_stairs_optimized_dp(self, n: int) -> int:
        """
        Approach 4: Optimized DP - Constant space
        
        Algorithm:
        1. Use only two variables to track previous two values
        2. Update variables iteratively
        3. This is the most space-efficient approach
        
        Time Complexity: O(n) - Single pass
        Space Complexity: O(1) - Only using two variables
        
        Analysis:
        - Pros: Optimal space complexity, very efficient
        - Cons: Slightly more complex logic
        """
        if n <= 2:
            return n
        
        prev1, prev2 = 1, 2  # Base cases for n=1 and n=2
        
        for i in range(3, n + 1):
            current = prev1 + prev2
            prev1, prev2 = prev2, current
        
        return prev2
    
    def climb_stairs_matrix_exponentiation(self, n: int) -> int:
        """
        Approach 5: Matrix Exponentiation - O(log n) time
        
        Algorithm:
        1. Use matrix multiplication to calculate Fibonacci
        2. [[1, 1], [1, 0]]^n gives us the nth Fibonacci number
        3. Use fast exponentiation for O(log n) time
        
        Time Complexity: O(log n) - Fast matrix exponentiation
        Space Complexity: O(1) - Constant space
        
        Analysis:
        - Pros: Very fast for large n
        - Cons: More complex implementation
        """
        if n <= 2:
            return n
        
        def matrix_multiply(a, b):
            return [
                [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
                [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]
            ]
        
        def matrix_power(matrix, power):
            if power == 0:
                return [[1, 0], [0, 1]]
            if power == 1:
                return matrix
            
            half = matrix_power(matrix, power // 2)
            squared = matrix_multiply(half, half)
            
            if power % 2 == 0:
                return squared
            else:
                return matrix_multiply(squared, matrix)
        
        # The matrix [[1, 1], [1, 0]]^n gives us F(n+1) and F(n)
        matrix = [[1, 1], [1, 0]]
        result = matrix_power(matrix, n)
        
        return result[0][0]
    
    def climb_stairs_brute_force(self, n: int) -> int:
        """
        Approach 6: Brute Force - Generate all possible combinations
        
        Algorithm:
        1. Generate all possible combinations of 1s and 2s that sum to n
        2. Count the number of valid combinations
        3. This is extremely inefficient but shows the concept
        
        Time Complexity: O(2^n) - Exponential
        Space Complexity: O(n) - Recursion stack
        
        Analysis:
        - Pros: Shows all possible combinations
        - Cons: Extremely inefficient, only for educational purposes
        """
        def generate_combinations(target, current_sum, current_path):
            if current_sum == target:
                return 1
            if current_sum > target:
                return 0
            
            # Try adding 1 step
            ways1 = generate_combinations(target, current_sum + 1, current_path + [1])
            # Try adding 2 steps
            ways2 = generate_combinations(target, current_sum + 2, current_path + [2])
            
            return ways1 + ways2
        
        return generate_combinations(n, 0, [])


# Testing and Benchmarking

def test_climbing_stairs():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "n": 1,
            "expected": 1,
            "description": "Single step"
        },
        {
            "n": 2,
            "expected": 2,
            "description": "Two steps"
        },
        {
            "n": 3,
            "expected": 3,
            "description": "Three steps"
        },
        {
            "n": 4,
            "expected": 5,
            "description": "Four steps"
        },
        {
            "n": 5,
            "expected": 8,
            "description": "Five steps"
        },
        {
            "n": 10,
            "expected": 89,
            "description": "Ten steps"
        }
    ]
    
    approaches = [
        ("Recursive", solution.climb_stairs_recursive),
        ("Memoization", solution.climb_stairs_memoization),
        ("DP Array", solution.climb_stairs_dp),
        ("Optimized DP", solution.climb_stairs_optimized_dp),
        ("Matrix Exponentiation", solution.climb_stairs_matrix_exponentiation),
        ("Brute Force", solution.climb_stairs_brute_force)
    ]
    
    print("Climbing Stairs - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: n = {test['n']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['n'])
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with large input"""
    
    solution = Solution()
    
    # Test with larger values
    n = 40
    
    print(f"\nBenchmarking with n = {n}")
    print("=" * 50)
    
    approaches = [
        ("Memoization", solution.climb_stairs_memoization),
        ("DP Array", solution.climb_stairs_dp),
        ("Optimized DP", solution.climb_stairs_optimized_dp),
        ("Matrix Exponentiation", solution.climb_stairs_matrix_exponentiation)
    ]
    
    # Skip recursive and brute force for large input as they're too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(n)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds (Result: {result})")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Recursive", "O(2^n)", "O(n)", "Simple but exponential"),
        ("Memoization", "O(n)", "O(n)", "Top-down DP"),
        ("DP Array", "O(n)", "O(n)", "Bottom-up DP"),
        ("Optimized DP", "O(n)", "O(1)", "Optimal space complexity"),
        ("Matrix Exponentiation", "O(log n)", "O(1)", "Fastest for large n"),
        ("Brute Force", "O(2^n)", "O(n)", "Educational only")
    ]
    
    print(f"{'Approach':<25} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<25} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Climbing Stairs - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_climbing_stairs()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the optimized DP approach for optimal performance.")
