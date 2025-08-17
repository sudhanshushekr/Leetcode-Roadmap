"""
Jump Game - LeetCode Problem 55
https://leetcode.com/problems/jump-game/

You are given an integer array nums. You are initially positioned at the array's first index, 
and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

Example 1:
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, 
which makes it impossible to reach the last index.

Constraints:
- 1 <= nums.length <= 10^4
- 0 <= nums[i] <= 10^5
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Jump Game
    """
    
    def canJump(self, nums: List[int]) -> bool:
        """
        Main method - returns the most efficient solution
        """
        return self.can_jump_greedy(nums)
    
    def can_jump_greedy(self, nums: List[int]) -> bool:
        """
        Approach 1: Greedy - Track maximum reachable position
        
        Algorithm:
        1. Keep track of the maximum position we can reach
        2. For each position, update max_reachable if we can go further
        3. If we can't reach current position, return False
        4. If we can reach or exceed last index, return True
        
        Time Complexity: O(n) - Single pass through array
        Space Complexity: O(1) - Only using a few variables
        
        Analysis:
        - Pros: Optimal time and space complexity, simple logic
        - Cons: Greedy approach may not be intuitive
        """
        max_reachable = 0
        
        for i in range(len(nums)):
            # If we can't reach current position
            if i > max_reachable:
                return False
            
            # Update maximum reachable position
            max_reachable = max(max_reachable, i + nums[i])
            
            # If we can reach the last index
            if max_reachable >= len(nums) - 1:
                return True
        
        return True
    
    def can_jump_dp(self, nums: List[int]) -> bool:
        """
        Approach 2: Dynamic Programming - Memoization
        
        Algorithm:
        1. Use DP to track if each position is reachable
        2. For each position, check if we can reach it from previous positions
        3. Use memoization to avoid recalculations
        
        Time Complexity: O(n²) - For each position, check all previous positions
        Space Complexity: O(n) - DP array
        
        Analysis:
        - Pros: Clear logic, handles all cases
        - Cons: Less efficient than greedy
        """
        n = len(nums)
        dp = [False] * n
        dp[0] = True
        
        for i in range(n):
            if dp[i]:
                # Mark all positions reachable from i
                for j in range(1, nums[i] + 1):
                    if i + j < n:
                        dp[i + j] = True
                    if i + j >= n - 1:
                        return True
        
        return dp[n - 1]
    
    def can_jump_dfs(self, nums: List[int]) -> bool:
        """
        Approach 3: Depth-First Search with Memoization
        
        Algorithm:
        1. Use DFS to explore all possible jump paths
        2. Use memoization to avoid revisiting states
        3. Return True if we can reach the last index
        
        Time Complexity: O(n²) - Each position can be visited multiple times
        Space Complexity: O(n) - Recursion stack + memoization
        
        Analysis:
        - Pros: Explores all possibilities
        - Cons: Less efficient, uses recursion
        """
        n = len(nums)
        memo = {}
        
        def dfs(position):
            if position in memo:
                return memo[position]
            
            if position >= n - 1:
                return True
            
            if nums[position] == 0:
                return False
            
            # Try all possible jumps from current position
            for jump in range(1, nums[position] + 1):
                if dfs(position + jump):
                    memo[position] = True
                    return True
            
            memo[position] = False
            return False
        
        return dfs(0)
    
    def can_jump_bfs(self, nums: List[int]) -> bool:
        """
        Approach 4: Breadth-First Search
        
        Algorithm:
        1. Use BFS to explore all reachable positions
        2. Add all positions reachable from current position to queue
        3. Use visited set to avoid revisiting
        
        Time Complexity: O(n²) - Each position can be visited multiple times
        Space Complexity: O(n) - Queue + visited set
        
        Analysis:
        - Pros: Explores level by level
        - Cons: Less efficient, uses extra space
        """
        from collections import deque
        
        n = len(nums)
        queue = deque([0])
        visited = {0}
        
        while queue:
            position = queue.popleft()
            
            if position >= n - 1:
                return True
            
            # Add all reachable positions to queue
            for jump in range(1, nums[position] + 1):
                next_pos = position + jump
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return False
    
    def can_jump_brute_force(self, nums: List[int]) -> bool:
        """
        Approach 5: Brute Force - Try all possible paths
        
        Algorithm:
        1. Recursively try all possible jump combinations
        2. No optimization, just explore all paths
        3. Very inefficient for large inputs
        
        Time Complexity: O(2^n) - Exponential
        Space Complexity: O(n) - Recursion stack
        
        Analysis:
        - Pros: Simple to understand
        - Cons: Extremely inefficient, only for educational purposes
        """
        def can_reach_end(position):
            if position >= len(nums) - 1:
                return True
            
            if nums[position] == 0:
                return False
            
            # Try all possible jumps
            for jump in range(1, nums[position] + 1):
                if can_reach_end(position + jump):
                    return True
            
            return False
        
        return can_reach_end(0)


# Testing and Benchmarking

def test_jump_game():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "nums": [2, 3, 1, 1, 4],
            "expected": True,
            "description": "Can reach end"
        },
        {
            "nums": [3, 2, 1, 0, 4],
            "expected": False,
            "description": "Cannot reach end"
        },
        {
            "nums": [0],
            "expected": True,
            "description": "Single element"
        },
        {
            "nums": [1, 0, 1, 0],
            "expected": False,
            "description": "Stuck at zero"
        },
        {
            "nums": [2, 0],
            "expected": True,
            "description": "Can jump over zero"
        },
        {
            "nums": [1, 1, 1, 1],
            "expected": True,
            "description": "All ones"
        }
    ]
    
    approaches = [
        ("Greedy", solution.can_jump_greedy),
        ("DP", solution.can_jump_dp),
        ("DFS", solution.can_jump_dfs),
        ("BFS", solution.can_jump_bfs),
        ("Brute Force", solution.can_jump_brute_force)
    ]
    
    print("Jump Game - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: nums = {test['nums']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['nums'].copy())
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with larger input"""
    
    solution = Solution()
    
    # Test with larger input
    nums = [2, 3, 1, 1, 4, 2, 1, 0, 1, 2, 3, 1, 1, 4] * 10  # Larger array
    
    print(f"\nBenchmarking with array of length {len(nums)}")
    print("=" * 50)
    
    approaches = [
        ("Greedy", solution.can_jump_greedy),
        ("DP", solution.can_jump_dp),
        ("DFS", solution.can_jump_dfs),
        ("BFS", solution.can_jump_bfs)
    ]
    
    # Skip brute force for large input as it's too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(nums.copy())
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds (Result: {result})")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Greedy", "O(n)", "O(1)", "Optimal time and space"),
        ("DP", "O(n²)", "O(n)", "Clear logic, less efficient"),
        ("DFS", "O(n²)", "O(n)", "Explores all paths"),
        ("BFS", "O(n²)", "O(n)", "Level by level exploration"),
        ("Brute Force", "O(2^n)", "O(n)", "Educational only")
    ]
    
    print(f"{'Approach':<15} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<15} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Jump Game - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_jump_game()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the greedy approach for optimal performance.")
