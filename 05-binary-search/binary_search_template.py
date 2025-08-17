"""
Binary Search Problems - NeetCode Style Template

This template provides a structured approach to solving binary search problems
with multiple solutions, complexity analysis, and clear explanations.

Common Patterns:
1. Standard Binary Search
2. Binary Search on Answer Space
3. Binary Search with Custom Condition
4. Binary Search on 2D Arrays
5. Binary Search with Duplicates
6. Binary Search on Rotated Arrays
"""

from typing import List, Optional
import time


class Solution:
    """
    Main solution class following NeetCode's structure
    """
    
    def problem_name(self, nums: List[int], target: int) -> int:
        """
        Problem: [Brief description of the problem]
        
        Example:
        Input: nums = [1, 2, 3, 4, 5], target = 3
        Output: 2
        
        Approach 1: Linear Search
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        
        # Approach 1: Linear Search Solution
        def linear_search_solution(nums: List[int], target: int) -> int:
            """
            Linear search approach - check each element
            """
            for i, num in enumerate(nums):
                if num == target:
                    return i
            return -1
        
        # Approach 2: Binary Search Solution
        def binary_search_solution(nums: List[int], target: int) -> int:
            """
            Binary search approach - efficient solution
            """
            left, right = 0, len(nums) - 1
            
            while left <= right:
                mid = (left + right) // 2
                
                if nums[mid] == target:
                    return mid
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return -1
        
        # Approach 3: Most Efficient Solution
        def most_efficient_solution(nums: List[int], target: int) -> int:
            """
            Most efficient approach - often involves mathematical insights
            """
            # Mathematical approach or advanced algorithm
            return result
        
        # Return the most efficient solution
        return binary_search_solution(nums, target)


# Common Binary Search Patterns

class BinarySearchPatterns:
    """Common patterns and techniques for binary search problems"""
    
    @staticmethod
    def binary_search_standard(nums: List[int], target: int) -> int:
        """
        Standard Binary Search - Find exact target
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    @staticmethod
    def binary_search_first_occurrence(nums: List[int], target: int) -> int:
        """
        Binary Search - Find first occurrence of target
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                result = mid
                right = mid - 1  # Continue searching left
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    @staticmethod
    def binary_search_last_occurrence(nums: List[int], target: int) -> int:
        """
        Binary Search - Find last occurrence of target
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                result = mid
                left = mid + 1  # Continue searching right
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    @staticmethod
    def binary_search_insert_position(nums: List[int], target: int) -> int:
        """
        Binary Search - Find insert position for target
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums)
        
        while left < right:
            mid = (left + right) // 2
            
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    @staticmethod
    def binary_search_rotated_array(nums: List[int], target: int) -> int:
        """
        Binary Search in Rotated Sorted Array
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                return mid
            
            # Check if left half is sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # Right half is sorted
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    @staticmethod
    def binary_search_answer_space(left: int, right: int, condition_func) -> int:
        """
        Binary Search on Answer Space
        Time: O(log n), Space: O(1)
        """
        while left < right:
            mid = (left + right) // 2
            
            if condition_func(mid):
                right = mid
            else:
                left = mid + 1
        
        return left
    
    @staticmethod
    def binary_search_2d_matrix(matrix: List[List[int]], target: int) -> bool:
        """
        Binary Search in 2D Matrix
        Time: O(log(m*n)), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        rows, cols = len(matrix), len(matrix[0])
        left, right = 0, rows * cols - 1
        
        while left <= right:
            mid = (left + right) // 2
            row, col = mid // cols, mid % cols
            
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return False


# Testing and Benchmarking

def test_solutions():
    """Test function to verify solutions work correctly"""
    
    # Test cases
    test_cases = [
        {
            "name": "Standard Binary Search",
            "input": ([1, 2, 3, 4, 5], 3),
            "expected": 2
        },
        {
            "name": "First Occurrence",
            "input": ([1, 2, 2, 2, 3], 2),
            "expected": 1
        },
        {
            "name": "Insert Position",
            "input": ([1, 3, 5, 6], 2),
            "expected": 1
        }
    ]
    
    patterns = BinarySearchPatterns()
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        
        if test['name'] == "Standard Binary Search":
            result = patterns.binary_search_standard(*test['input'])
        elif test['name'] == "First Occurrence":
            result = patterns.binary_search_first_occurrence(*test['input'])
        elif test['name'] == "Insert Position":
            result = patterns.binary_search_insert_position(*test['input'])
        
        print(f"Input: {test['input']}")
        print(f"Expected: {test['expected']}")
        print(f"Result: {result}")
        print(f"Pass: {result == test['expected']}")


def benchmark_solutions():
    """Benchmark different approaches to compare performance"""
    
    # Example: Compare different approaches for the same problem
    nums = list(range(1000000))
    target = 500000
    
    print("Benchmarking Binary Search approaches:")
    
    # Approach 1: Linear Search
    start_time = time.time()
    # linear_result = linear_search(nums, target)
    linear_time = time.time() - start_time
    
    # Approach 2: Binary Search
    start_time = time.time()
    binary_result = BinarySearchPatterns.binary_search_standard(nums, target)
    binary_time = time.time() - start_time
    
    print(f"Linear Search: {linear_time:.6f} seconds")
    print(f"Binary Search: {binary_time:.6f} seconds")
    print(f"Speedup: {linear_time / binary_time:.2f}x")


if __name__ == "__main__":
    print("Binary Search Solutions - NeetCode Style")
    print("=" * 50)
    
    # Run tests
    test_solutions()
    
    # Run benchmarks
    print("\n" + "=" * 50)
    benchmark_solutions()
    
    print("\nTemplate ready for use! Add your specific problem solutions here.")
