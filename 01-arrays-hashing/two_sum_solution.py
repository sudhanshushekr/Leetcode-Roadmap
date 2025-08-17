"""
Two Sum - LeetCode Problem 1
https://leetcode.com/problems/two-sum/

Given an array of integers nums and an integer target, return indices of the two numbers 
such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use 
the same element twice.

You can return the answer in any order.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]

Constraints:
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
- Only one valid answer exists.
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Two Sum
    """
    
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        Main method - returns the most efficient solution
        """
        return self.two_sum_hash_map(nums, target)
    
    def two_sum_brute_force(self, nums: List[int], target: int) -> List[int]:
        """
        Approach 1: Brute Force - Check all pairs
        
        Algorithm:
        1. Use nested loops to check every possible pair
        2. If sum equals target, return the indices
        
        Time Complexity: O(n²) - We check n*(n-1)/2 pairs
        Space Complexity: O(1) - Only using constant extra space
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Very inefficient for large arrays
        """
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] + nums[j] == target:
                    return [i, j]
        
        return []  # No solution found
    
    def two_sum_hash_map(self, nums: List[int], target: int) -> List[int]:
        """
        Approach 2: Hash Map - One Pass Solution
        
        Algorithm:
        1. Use a hash map to store numbers we've seen
        2. For each number, check if (target - current_number) exists in map
        3. If found, return the indices
        4. If not found, add current number and its index to map
        
        Time Complexity: O(n) - Single pass through the array
        Space Complexity: O(n) - Hash map can store up to n elements
        
        Analysis:
        - Pros: Optimal time complexity, handles duplicates correctly
        - Cons: Uses extra space for hash map
        """
        seen = {}  # val -> index
        
        for i, num in enumerate(nums):
            complement = target - num
            
            # Check if complement exists in our hash map
            if complement in seen:
                return [seen[complement], i]
            
            # Add current number and its index to hash map
            seen[num] = i
        
        return []  # No solution found
    
    def two_sum_two_pass_hash_map(self, nums: List[int], target: int) -> List[int]:
        """
        Approach 3: Two Pass Hash Map
        
        Algorithm:
        1. First pass: Build hash map with all numbers and their indices
        2. Second pass: For each number, check if complement exists
        
        Time Complexity: O(n) - Two passes through the array
        Space Complexity: O(n) - Hash map stores all elements
        
        Analysis:
        - Pros: Clear separation of building map and searching
        - Cons: Requires two passes, slightly more complex
        """
        # First pass: Build hash map
        num_map = {}
        for i, num in enumerate(nums):
            num_map[num] = i
        
        # Second pass: Find complement
        for i, num in enumerate(nums):
            complement = target - num
            
            # Check if complement exists and it's not the same element
            if complement in num_map and num_map[complement] != i:
                return [i, num_map[complement]]
        
        return []  # No solution found
    
    def two_sum_sorting_approach(self, nums: List[int], target: int) -> List[int]:
        """
        Approach 4: Sorting + Two Pointers (Returns values, not indices)
        
        Note: This approach returns the actual values, not indices,
        so it's not suitable for the original problem but shows the technique.
        
        Algorithm:
        1. Sort the array
        2. Use two pointers (left and right)
        3. If sum < target, move left pointer right
        4. If sum > target, move right pointer left
        5. If sum == target, return the values
        
        Time Complexity: O(n log n) - Due to sorting
        Space Complexity: O(1) - Only using pointers
        
        Analysis:
        - Pros: Space efficient, good for finding values
        - Cons: Doesn't preserve original indices, requires sorting
        """
        # Create list of (value, index) pairs to preserve indices
        nums_with_indices = [(nums[i], i) for i in range(len(nums))]
        nums_with_indices.sort()  # Sort by values
        
        left, right = 0, len(nums_with_indices) - 1
        
        while left < right:
            current_sum = nums_with_indices[left][0] + nums_with_indices[right][0]
            
            if current_sum == target:
                return [nums_with_indices[left][1], nums_with_indices[right][1]]
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []  # No solution found


# Testing and Benchmarking

def test_two_sum():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "nums": [2, 7, 11, 15],
            "target": 9,
            "expected": [0, 1],
            "description": "Basic case"
        },
        {
            "nums": [3, 2, 4],
            "target": 6,
            "expected": [1, 2],
            "description": "Target not at beginning"
        },
        {
            "nums": [3, 3],
            "target": 6,
            "expected": [0, 1],
            "description": "Same numbers"
        },
        {
            "nums": [1, 5, 8, 10, 13],
            "target": 18,
            "expected": [2, 4],  # 8 + 10 = 18
            "description": "Larger array"
        },
        {
            "nums": [-1, -2, -3, -4, -5],
            "target": -8,
            "expected": [2, 4],
            "description": "Negative numbers"
        }
    ]
    
    approaches = [
        ("Brute Force", solution.two_sum_brute_force),
        ("Hash Map (One Pass)", solution.two_sum_hash_map),
        ("Hash Map (Two Pass)", solution.two_sum_two_pass_hash_map),
        ("Sorting + Two Pointers", solution.two_sum_sorting_approach)
    ]
    
    print("Two Sum - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: nums = {test['nums']}, target = {test['target']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['nums'], test['target'])
                # Sort results for comparison since order doesn't matter
                result_sorted = sorted(result)
                expected_sorted = sorted(test['expected'])
                
                passed = result_sorted == expected_sorted
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with large input"""
    
    solution = Solution()
    
    # Create large test case
    n = 10000
    nums = list(range(n))
    target = 2 * n - 1  # Will find indices n-2 and n-1
    
    print(f"\nBenchmarking with array of size {n}")
    print("=" * 50)
    
    approaches = [
        ("Hash Map (One Pass)", solution.two_sum_hash_map),
        ("Hash Map (Two Pass)", solution.two_sum_two_pass_hash_map),
        ("Sorting + Two Pointers", solution.two_sum_sorting_approach)
    ]
    
    # Skip brute force for large input as it's too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(nums, target)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds")
        print(f"  Result: {result}")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Brute Force", "O(n²)", "O(1)", "Simple but very slow"),
        ("Hash Map (One Pass)", "O(n)", "O(n)", "Optimal time, good for most cases"),
        ("Hash Map (Two Pass)", "O(n)", "O(n)", "Clear separation, slightly more complex"),
        ("Sorting + Two Pointers", "O(n log n)", "O(1)", "Space efficient, but slower")
    ]
    
    print(f"{'Approach':<25} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<25} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Two Sum - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_two_sum()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the hash map approach for optimal performance.")
