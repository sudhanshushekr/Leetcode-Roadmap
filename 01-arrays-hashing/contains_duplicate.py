"""
Contains Duplicate - LeetCode Problem 217
https://leetcode.com/problems/contains-duplicate/

Given an integer array nums, return true if any value appears at least twice in the array, 
and return false if every element is distinct.

Example 1:
Input: nums = [1,2,3,1]
Output: true

Example 2:
Input: nums = [1,2,3,4]
Output: false

Example 3:
Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true

Constraints:
- 1 <= nums.length <= 10^5
- -10^9 <= nums[i] <= 10^9
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Contains Duplicate
    """
    
    def containsDuplicate(self, nums: List[int]) -> bool:
        """
        Main method - returns the most efficient solution
        """
        return self.contains_duplicate_hash_set(nums)
    
    def contains_duplicate_brute_force(self, nums: List[int]) -> bool:
        """
        Approach 1: Brute Force - Check all pairs
        
        Algorithm:
        1. Use nested loops to compare each element with all others
        2. If any duplicate is found, return True
        3. If no duplicates found after checking all pairs, return False
        
        Time Complexity: O(n²) - We compare each element with n-1 other elements
        Space Complexity: O(1) - Only using constant extra space
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Very inefficient for large arrays
        """
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] == nums[j]:
                    return True
        
        return False
    
    def contains_duplicate_sorting(self, nums: List[int]) -> bool:
        """
        Approach 2: Sorting - Check adjacent elements
        
        Algorithm:
        1. Sort the array
        2. Check if any adjacent elements are equal
        3. If found, return True, else return False
        
        Time Complexity: O(n log n) - Due to sorting
        Space Complexity: O(1) - In-place sorting (modifies input)
        
        Analysis:
        - Pros: Space efficient, simple logic after sorting
        - Cons: Modifies input array, slower than hash set approach
        """
        nums.sort()  # Modifies the input array
        
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                return True
        
        return False
    
    def contains_duplicate_hash_set(self, nums: List[int]) -> bool:
        """
        Approach 3: Hash Set - Track seen elements
        
        Algorithm:
        1. Use a hash set to track elements we've seen
        2. For each element, check if it's already in the set
        3. If found, return True, else add to set and continue
        4. If no duplicates found, return False
        
        Time Complexity: O(n) - Single pass through the array
        Space Complexity: O(n) - Hash set can store up to n elements
        
        Analysis:
        - Pros: Optimal time complexity, doesn't modify input
        - Cons: Uses extra space for hash set
        """
        seen = set()
        
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        
        return False
    
    def contains_duplicate_hash_map(self, nums: List[int]) -> bool:
        """
        Approach 4: Hash Map - Count frequencies
        
        Algorithm:
        1. Use a hash map to count frequency of each element
        2. If any element has frequency > 1, return True
        3. If all elements have frequency 1, return False
        
        Time Complexity: O(n) - Single pass through the array
        Space Complexity: O(n) - Hash map can store up to n elements
        
        Analysis:
        - Pros: Can be extended to find all duplicates
        - Cons: More complex than hash set approach
        """
        from collections import Counter
        
        count = Counter(nums)
        
        for freq in count.values():
            if freq > 1:
                return True
        
        return False


# Testing and Benchmarking

def test_contains_duplicate():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "nums": [1, 2, 3, 1],
            "expected": True,
            "description": "Has duplicates"
        },
        {
            "nums": [1, 2, 3, 4],
            "expected": False,
            "description": "No duplicates"
        },
        {
            "nums": [1, 1, 1, 3, 3, 4, 3, 2, 4, 2],
            "expected": True,
            "description": "Multiple duplicates"
        },
        {
            "nums": [1],
            "expected": False,
            "description": "Single element"
        },
        {
            "nums": [1, 1],
            "expected": True,
            "description": "Two same elements"
        },
        {
            "nums": [],
            "expected": False,
            "description": "Empty array"
        },
        {
            "nums": [-1, -2, -1, 0, 1],
            "expected": True,
            "description": "Negative numbers with duplicates"
        }
    ]
    
    approaches = [
        ("Brute Force", solution.contains_duplicate_brute_force),
        ("Sorting", solution.contains_duplicate_sorting),
        ("Hash Set", solution.contains_duplicate_hash_set),
        ("Hash Map", solution.contains_duplicate_hash_map)
    ]
    
    print("Contains Duplicate - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: nums = {test['nums']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                # For sorting approach, we need to copy the array since it modifies input
                if approach_name == "Sorting":
                    nums_copy = test['nums'].copy()
                    result = approach_func(nums_copy)
                else:
                    result = approach_func(test['nums'])
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with large input"""
    
    solution = Solution()
    
    # Create large test case with duplicates
    n = 10000
    nums_with_duplicates = list(range(n)) + [n//2]  # Add a duplicate
    nums_without_duplicates = list(range(n))
    
    print(f"\nBenchmarking with array of size {n}")
    print("=" * 50)
    
    approaches = [
        ("Hash Set", solution.contains_duplicate_hash_set),
        ("Hash Map", solution.contains_duplicate_hash_map),
        ("Sorting", solution.contains_duplicate_sorting)
    ]
    
    # Skip brute force for large input as it's too slow
    for approach_name, approach_func in approaches:
        # Test with duplicates
        start_time = time.time()
        result_with_dups = approach_func(nums_with_duplicates.copy())
        time_with_dups = time.time() - start_time
        
        # Test without duplicates
        start_time = time.time()
        result_without_dups = approach_func(nums_without_duplicates.copy())
        time_without_dups = time.time() - start_time
        
        print(f"{approach_name}:")
        print(f"  With duplicates: {time_with_dups:.6f} seconds (Result: {result_with_dups})")
        print(f"  Without duplicates: {time_without_dups:.6f} seconds (Result: {result_without_dups})")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Brute Force", "O(n²)", "O(1)", "Simple but very slow"),
        ("Sorting", "O(n log n)", "O(1)", "Space efficient, modifies input"),
        ("Hash Set", "O(n)", "O(n)", "Optimal time, most practical"),
        ("Hash Map", "O(n)", "O(n)", "Can be extended for frequency analysis")
    ]
    
    print(f"{'Approach':<15} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<15} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Contains Duplicate - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_contains_duplicate()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the hash set approach for optimal performance.")
