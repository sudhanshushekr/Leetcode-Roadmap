"""
Two Sum II - Input Array Is Sorted - LeetCode Problem 167
https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/

Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, 
find two numbers such that they add up to a specific target number. Let these two numbers 
be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer 
array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the 
same element twice.

Your solution must use only constant extra space.

Example 1:
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].

Example 2:
Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore index1 = 1, index2 = 3. We return [1, 3].

Example 3:
Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore index1 = 1, index2 = 2. We return [1, 2].

Constraints:
- 2 <= numbers.length <= 3 * 10^4
- -1000 <= numbers[i] <= 1000
- numbers is sorted in non-decreasing order.
- -1000 <= target <= 1000
- The tests are generated such that there is exactly one solution.
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Two Sum II
    """
    
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        """
        Main method - returns the most efficient solution
        """
        return self.two_sum_two_pointers(numbers, target)
    
    def two_sum_brute_force(self, numbers: List[int], target: int) -> List[int]:
        """
        Approach 1: Brute Force - Check all pairs
        
        Algorithm:
        1. Use nested loops to check every possible pair
        2. If sum equals target, return the indices (1-indexed)
        
        Time Complexity: O(n²) - We check n*(n-1)/2 pairs
        Space Complexity: O(1) - Only using constant extra space
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Very inefficient for large arrays
        """
        n = len(numbers)
        
        for i in range(n):
            for j in range(i + 1, n):
                if numbers[i] + numbers[j] == target:
                    return [i + 1, j + 1]  # 1-indexed
        
        return []
    
    def two_sum_binary_search(self, numbers: List[int], target: int) -> List[int]:
        """
        Approach 2: Binary Search - For each element, search for complement
        
        Algorithm:
        1. For each element, calculate the complement (target - current)
        2. Use binary search to find the complement in the remaining array
        3. If found, return the indices
        
        Time Complexity: O(n log n) - For each element, binary search takes O(log n)
        Space Complexity: O(1) - Only using constant extra space
        
        Analysis:
        - Pros: More efficient than brute force
        - Cons: Still not optimal, binary search overhead
        """
        n = len(numbers)
        
        for i in range(n):
            complement = target - numbers[i]
            
            # Binary search for complement in the remaining array
            left, right = i + 1, n - 1
            
            while left <= right:
                mid = (left + right) // 2
                
                if numbers[mid] == complement:
                    return [i + 1, mid + 1]  # 1-indexed
                elif numbers[mid] < complement:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return []
    
    def two_sum_two_pointers(self, numbers: List[int], target: int) -> List[int]:
        """
        Approach 3: Two Pointers - Optimal Solution
        
        Algorithm:
        1. Use two pointers, one at start and one at end
        2. If sum < target, move left pointer right
        3. If sum > target, move right pointer left
        4. If sum == target, return the indices
        
        Time Complexity: O(n) - Single pass through the array
        Space Complexity: O(1) - Only using two pointers
        
        Analysis:
        - Pros: Optimal time complexity, very efficient
        - Cons: Requires sorted array (which is given)
        """
        left, right = 0, len(numbers) - 1
        
        while left < right:
            current_sum = numbers[left] + numbers[right]
            
            if current_sum == target:
                return [left + 1, right + 1]  # 1-indexed
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []
    
    def two_sum_hash_map(self, numbers: List[int], target: int) -> List[int]:
        """
        Approach 4: Hash Map - Same as Two Sum I
        
        Algorithm:
        1. Use hash map to store numbers we've seen
        2. For each number, check if (target - current_number) exists in map
        3. If found, return the indices
        
        Time Complexity: O(n) - Single pass through the array
        Space Complexity: O(n) - Hash map can store up to n elements
        
        Analysis:
        - Pros: Works for any array (sorted or not)
        - Cons: Uses extra space, not taking advantage of sorted property
        """
        seen = {}  # val -> index
        
        for i, num in enumerate(numbers):
            complement = target - num
            
            if complement in seen:
                return [seen[complement] + 1, i + 1]  # 1-indexed
            
            seen[num] = i
        
        return []
    
    def two_sum_linear_search(self, numbers: List[int], target: int) -> List[int]:
        """
        Approach 5: Linear Search - For each element, search linearly for complement
        
        Algorithm:
        1. For each element, search linearly for its complement
        2. Since array is sorted, we can stop early if we exceed target
        
        Time Complexity: O(n²) - For each element, linear search
        Space Complexity: O(1) - Only using constant extra space
        
        Analysis:
        - Pros: Simple implementation
        - Cons: Inefficient, quadratic time complexity
        """
        n = len(numbers)
        
        for i in range(n):
            complement = target - numbers[i]
            
            # Linear search for complement
            for j in range(i + 1, n):
                if numbers[j] == complement:
                    return [i + 1, j + 1]  # 1-indexed
                elif numbers[j] > complement:
                    break  # Early termination since array is sorted
        
        return []


# Testing and Benchmarking

def test_two_sum_ii():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "numbers": [2, 7, 11, 15],
            "target": 9,
            "expected": [1, 2],
            "description": "Basic case"
        },
        {
            "numbers": [2, 3, 4],
            "target": 6,
            "expected": [1, 3],
            "description": "Target not at beginning"
        },
        {
            "numbers": [-1, 0],
            "target": -1,
            "expected": [1, 2],
            "description": "Negative numbers"
        },
        {
            "numbers": [1, 2, 3, 4, 5],
            "target": 9,
            "expected": [4, 5],
            "description": "Larger array"
        },
        {
            "numbers": [0, 0, 3, 4],
            "target": 0,
            "expected": [1, 2],
            "description": "Same numbers"
        },
        {
            "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": 19,
            "expected": [9, 10],
            "description": "Large array"
        }
    ]
    
    approaches = [
        ("Brute Force", solution.two_sum_brute_force),
        ("Binary Search", solution.two_sum_binary_search),
        ("Two Pointers", solution.two_sum_two_pointers),
        ("Hash Map", solution.two_sum_hash_map),
        ("Linear Search", solution.two_sum_linear_search)
    ]
    
    print("Two Sum II - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: numbers = {test['numbers']}, target = {test['target']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['numbers'], test['target'])
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with large input"""
    
    solution = Solution()
    
    # Create large test case
    n = 10000
    numbers = list(range(1, n + 1))  # Sorted array
    target = 2 * n - 1  # Will find indices n-1 and n
    
    print(f"\nBenchmarking with array of size {n}")
    print("=" * 50)
    
    approaches = [
        ("Two Pointers", solution.two_sum_two_pointers),
        ("Hash Map", solution.two_sum_hash_map),
        ("Binary Search", solution.two_sum_binary_search)
    ]
    
    # Skip brute force and linear search for large input as they're too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(numbers, target)
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
        ("Binary Search", "O(n log n)", "O(1)", "Better than brute force"),
        ("Two Pointers", "O(n)", "O(1)", "Optimal for sorted arrays"),
        ("Hash Map", "O(n)", "O(n)", "Works for any array"),
        ("Linear Search", "O(n²)", "O(1)", "Inefficient")
    ]
    
    print(f"{'Approach':<20} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<20} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Two Sum II - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_two_sum_ii()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the two pointers approach for optimal performance.")
