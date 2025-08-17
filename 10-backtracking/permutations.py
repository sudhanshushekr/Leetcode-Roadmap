"""
Permutations - LeetCode Problem 46
https://leetcode.com/problems/permutations/

Given an array nums of distinct integers, return all the possible permutations. 
You can return the answer in any order.

Example 1:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Example 2:
Input: nums = [0,1]
Output: [[0,1],[1,0]]

Example 3:
Input: nums = [1]
Output: [[1]]

Constraints:
- 1 <= nums.length <= 6
- -10 <= nums[i] <= 10
- All the integers of nums are unique.
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Permutations
    """
    
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        Main method - returns the most efficient solution
        """
        return self.permute_backtracking(nums)
    
    def permute_backtracking(self, nums: List[int]) -> List[List[int]]:
        """
        Approach 1: Backtracking - Most Efficient
        
        Algorithm:
        1. Use backtracking to generate all permutations
        2. Swap elements to generate different arrangements
        3. Use a visited array or swap in-place
        
        Time Complexity: O(n!) - n! permutations
        Space Complexity: O(n) - Recursion stack + visited array
        
        Analysis:
        - Pros: Optimal time complexity, clear logic
        - Cons: Uses recursion stack space
        """
        def backtrack(start):
            if start == len(nums):
                result.append(nums[:])
                return
            
            for i in range(start, len(nums)):
                # Swap elements
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                # Backtrack
                nums[start], nums[i] = nums[i], nums[start]
        
        result = []
        backtrack(0)
        return result
    
    def permute_visited_array(self, nums: List[int]) -> List[List[int]]:
        """
        Approach 2: Backtracking with Visited Array
        
        Algorithm:
        1. Use a visited array to track used elements
        2. Build permutation one element at a time
        3. Mark elements as visited/unvisited
        
        Time Complexity: O(n!) - n! permutations
        Space Complexity: O(n) - Visited array + recursion stack
        
        Analysis:
        - Pros: Clear logic, easy to understand
        - Cons: Uses extra space for visited array
        """
        def backtrack(path, visited):
            if len(path) == len(nums):
                result.append(path[:])
                return
            
            for i in range(len(nums)):
                if not visited[i]:
                    visited[i] = True
                    path.append(nums[i])
                    backtrack(path, visited)
                    path.pop()
                    visited[i] = False
        
        result = []
        visited = [False] * len(nums)
        backtrack([], visited)
        return result
    
    def permute_iterative(self, nums: List[int]) -> List[List[int]]:
        """
        Approach 3: Iterative - Build permutations level by level
        
        Algorithm:
        1. Start with empty permutation
        2. For each number, insert it at all possible positions
        3. Build permutations iteratively
        
        Time Complexity: O(n!) - n! permutations
        Space Complexity: O(n!) - Store all permutations
        
        Analysis:
        - Pros: No recursion, iterative approach
        - Cons: Uses more space, less intuitive
        """
        if not nums:
            return []
        
        result = [[]]
        
        for num in nums:
            new_result = []
            for perm in result:
                for i in range(len(perm) + 1):
                    new_perm = perm[:i] + [num] + perm[i:]
                    new_result.append(new_perm)
            result = new_result
        
        return result
    
    def permute_heap_algorithm(self, nums: List[int]) -> List[List[int]]:
        """
        Approach 4: Heap's Algorithm - Generate permutations by adjacent swaps
        
        Algorithm:
        1. Use Heap's algorithm for generating permutations
        2. Generate permutations by swapping adjacent elements
        3. More efficient than standard backtracking
        
        Time Complexity: O(n!) - n! permutations
        Space Complexity: O(n) - Recursion stack
        
        Analysis:
        - Pros: Efficient algorithm, minimal swaps
        - Cons: More complex implementation
        """
        def generate_permutations(n):
            if n == 1:
                result.append(nums[:])
                return
            
            for i in range(n):
                generate_permutations(n - 1)
                
                # Swap based on whether n is odd or even
                if n % 2 == 1:
                    nums[0], nums[n - 1] = nums[n - 1], nums[0]
                else:
                    nums[i], nums[n - 1] = nums[n - 1], nums[i]
        
        result = []
        generate_permutations(len(nums))
        return result
    
    def permute_brute_force(self, nums: List[int]) -> List[List[int]]:
        """
        Approach 5: Brute Force - Generate all possible arrangements
        
        Algorithm:
        1. Generate all possible arrangements
        2. Check if each arrangement is a valid permutation
        3. Very inefficient approach
        
        Time Complexity: O(n^n) - Generate all possible arrangements
        Space Complexity: O(n^n) - Store all arrangements
        
        Analysis:
        - Pros: Simple to understand
        - Cons: Extremely inefficient, only for educational purposes
        """
        from itertools import permutations
        return list(permutations(nums))


# Testing and Benchmarking

def test_permutations():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "nums": [1, 2, 3],
            "expected": [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]],
            "description": "Three elements"
        },
        {
            "nums": [0, 1],
            "expected": [[0, 1], [1, 0]],
            "description": "Two elements"
        },
        {
            "nums": [1],
            "expected": [[1]],
            "description": "Single element"
        },
        {
            "nums": [1, 2],
            "expected": [[1, 2], [2, 1]],
            "description": "Two elements"
        }
    ]
    
    approaches = [
        ("Backtracking", solution.permute_backtracking),
        ("Visited Array", solution.permute_visited_array),
        ("Iterative", solution.permute_iterative),
        ("Heap's Algorithm", solution.permute_heap_algorithm)
    ]
    
    print("Permutations - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: nums = {test['nums']}")
        print(f"Expected count: {len(test['expected'])}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['nums'].copy())
                
                # Sort both result and expected for comparison
                result_sorted = sorted([tuple(perm) for perm in result])
                expected_sorted = sorted([tuple(perm) for perm in test['expected']])
                
                passed = result_sorted == expected_sorted
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {len(result)} permutations {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with larger input"""
    
    solution = Solution()
    
    # Test with larger input
    nums = [1, 2, 3, 4]  # 4! = 24 permutations
    
    print(f"\nBenchmarking with nums = {nums} (24 permutations)")
    print("=" * 50)
    
    approaches = [
        ("Backtracking", solution.permute_backtracking),
        ("Visited Array", solution.permute_visited_array),
        ("Iterative", solution.permute_iterative),
        ("Heap's Algorithm", solution.permute_heap_algorithm)
    ]
    
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(nums.copy())
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds ({len(result)} permutations)")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Backtracking", "O(n!)", "O(n)", "Optimal, in-place swaps"),
        ("Visited Array", "O(n!)", "O(n)", "Clear logic, extra space"),
        ("Iterative", "O(n!)", "O(n!)", "No recursion, more space"),
        ("Heap's Algorithm", "O(n!)", "O(n)", "Efficient swaps"),
        ("Brute Force", "O(n^n)", "O(n^n)", "Educational only")
    ]
    
    print(f"{'Approach':<20} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<20} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Permutations - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_permutations()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the backtracking approach for optimal performance.")
