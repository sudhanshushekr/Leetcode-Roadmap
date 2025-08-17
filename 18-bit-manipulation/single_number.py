"""
Single Number - LeetCode Problem 136
https://leetcode.com/problems/single-number/

Given a non-empty array of integers nums, every element appears twice except for one. 
Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

Example 1:
Input: nums = [2,2,1]
Output: 1

Example 2:
Input: nums = [4,1,2,1,2]
Output: 4

Example 3:
Input: nums = [1]
Output: 1

Constraints:
- 1 <= nums.length <= 3 * 10^4
- -3 * 10^4 <= nums[i] <= 3 * 10^4
- Each element in the array appears twice except for one element which appears only once.
"""

from typing import List
import time
from collections import Counter


class Solution:
    """
    Solution class with multiple approaches to solve Single Number
    """
    
    def singleNumber(self, nums: List[int]) -> int:
        """
        Main method - returns the most efficient solution
        """
        return self.single_number_xor(nums)
    
    def single_number_xor(self, nums: List[int]) -> int:
        """
        Approach 1: XOR Operation - Most Efficient
        
        Algorithm:
        1. Use XOR operation: a ^ a = 0, a ^ 0 = a
        2. XOR all numbers in the array
        3. The result is the single number
        
        Time Complexity: O(n) - Single pass through array
        Space Complexity: O(1) - Only one variable
        
        Analysis:
        - Pros: Optimal time and space complexity, elegant solution
        - Cons: Requires understanding of XOR properties
        """
        result = 0
        for num in nums:
            result ^= num
        return result
    
    def single_number_hash_set(self, nums: List[int]) -> int:
        """
        Approach 2: Hash Set - Track seen numbers
        
        Algorithm:
        1. Use a hash set to track seen numbers
        2. Add number if not seen, remove if seen
        3. The remaining number is the single one
        
        Time Complexity: O(n) - Single pass through array
        Space Complexity: O(n) - Hash set can store up to n/2 elements
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Uses extra space
        """
        seen = set()
        
        for num in nums:
            if num in seen:
                seen.remove(num)
            else:
                seen.add(num)
        
        return seen.pop()
    
    def single_number_counter(self, nums: List[int]) -> int:
        """
        Approach 3: Counter - Count occurrences
        
        Algorithm:
        1. Use Counter to count occurrences of each number
        2. Find the number with count 1
        
        Time Complexity: O(n) - Count all elements
        Space Complexity: O(n) - Counter stores all unique elements
        
        Analysis:
        - Pros: Clear logic, uses built-in Counter
        - Cons: Uses extra space
        """
        count = Counter(nums)
        
        for num, freq in count.items():
            if freq == 1:
                return num
        
        return -1  # Should not reach here
    
    def single_number_sorting(self, nums: List[int]) -> int:
        """
        Approach 4: Sorting - Compare adjacent elements
        
        Algorithm:
        1. Sort the array
        2. Compare adjacent elements
        3. The single number will be the one that doesn't match its neighbors
        
        Time Complexity: O(n log n) - Sorting dominates
        Space Complexity: O(1) - In-place sorting
        
        Analysis:
        - Pros: Simple logic after sorting
        - Cons: Not optimal time complexity
        """
        nums.sort()
        
        for i in range(0, len(nums) - 1, 2):
            if nums[i] != nums[i + 1]:
                return nums[i]
        
        # If we reach here, the single number is the last element
        return nums[-1]
    
    def single_number_math(self, nums: List[int]) -> int:
        """
        Approach 5: Mathematical - Sum of unique elements
        
        Algorithm:
        1. Calculate sum of all unique elements * 2
        2. Subtract sum of all elements
        3. Result is the single number
        
        Time Complexity: O(n) - Two passes through array
        Space Complexity: O(n) - Set of unique elements
        
        Analysis:
        - Pros: Mathematical approach
        - Cons: Uses extra space, two passes
        """
        unique_nums = set(nums)
        return 2 * sum(unique_nums) - sum(nums)
    
    def single_number_brute_force(self, nums: List[int]) -> int:
        """
        Approach 6: Brute Force - Check each element
        
        Algorithm:
        1. For each element, check if it appears elsewhere
        2. Return the element that doesn't appear twice
        
        Time Complexity: O(n²) - For each element, check all others
        Space Complexity: O(1) - No extra space
        
        Analysis:
        - Pros: Simple logic
        - Cons: Very inefficient, only for educational purposes
        """
        for i in range(len(nums)):
            found = False
            for j in range(len(nums)):
                if i != j and nums[i] == nums[j]:
                    found = True
                    break
            if not found:
                return nums[i]
        
        return -1  # Should not reach here


# Testing and Benchmarking

def test_single_number():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "nums": [2, 2, 1],
            "expected": 1,
            "description": "Single number at end"
        },
        {
            "nums": [4, 1, 2, 1, 2],
            "expected": 4,
            "description": "Single number at beginning"
        },
        {
            "nums": [1],
            "expected": 1,
            "description": "Single element"
        },
        {
            "nums": [1, 2, 3, 4, 5, 1, 2, 3, 4],
            "expected": 5,
            "description": "Single number in middle"
        },
        {
            "nums": [7, 3, 5, 3, 5],
            "expected": 7,
            "description": "Single number at beginning"
        }
    ]
    
    approaches = [
        ("XOR", solution.single_number_xor),
        ("Hash Set", solution.single_number_hash_set),
        ("Counter", solution.single_number_counter),
        ("Sorting", solution.single_number_sorting),
        ("Math", solution.single_number_math),
        ("Brute Force", solution.single_number_brute_force)
    ]
    
    print("Single Number - Testing All Approaches")
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
    n = 10000
    nums = list(range(n // 2)) * 2 + [n]  # All numbers appear twice except n
    
    print(f"\nBenchmarking with array of length {len(nums)}")
    print("=" * 50)
    
    approaches = [
        ("XOR", solution.single_number_xor),
        ("Hash Set", solution.single_number_hash_set),
        ("Counter", solution.single_number_counter),
        ("Sorting", solution.single_number_sorting),
        ("Math", solution.single_number_math)
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
        ("XOR", "O(n)", "O(1)", "Optimal time and space"),
        ("Hash Set", "O(n)", "O(n)", "Simple logic"),
        ("Counter", "O(n)", "O(n)", "Built-in Counter"),
        ("Sorting", "O(n log n)", "O(1)", "In-place sorting"),
        ("Math", "O(n)", "O(n)", "Mathematical approach"),
        ("Brute Force", "O(n²)", "O(1)", "Educational only")
    ]
    
    print(f"{'Approach':<15} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<15} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Single Number - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_single_number()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the XOR approach for optimal performance.")
