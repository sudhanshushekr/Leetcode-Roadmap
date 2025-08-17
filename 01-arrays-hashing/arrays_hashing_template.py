"""
Arrays & Hashing Problems - NeetCode Style Template

This template provides a structured approach to solving arrays and hashing problems
with multiple solutions, complexity analysis, and clear explanations.

Common Patterns:
1. Two Pointers
2. Sliding Window
3. Hash Set/Map
4. Sorting + Two Pointers
5. Prefix Sum
6. Kadane's Algorithm
"""

from typing import List, Dict, Set, Optional
from collections import defaultdict, Counter
import time


class Solution:
    """
    Main solution class following NeetCode's structure
    """
    
    def problem_name(self, nums: List[int]) -> int:
        """
        Problem: [Brief description of the problem]
        
        Example:
        Input: nums = [1, 2, 3, 4]
        Output: 10
        
        Approach 1: Brute Force
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        
        # Approach 1: Brute Force Solution
        def brute_force_solution(nums: List[int]) -> int:
            """
            Brute force approach - usually the first solution that comes to mind
            """
            result = 0
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    # Your logic here
                    pass
            return result
        
        # Approach 2: Optimized Solution
        def optimized_solution(nums: List[int]) -> int:
            """
            Optimized approach using hash set/map or other efficient data structure
            """
            # Initialize data structures
            seen = set()  # or dict, Counter, etc.
            
            # Main logic
            for num in nums:
                # Your optimized logic here
                pass
            
            return result
        
        # Approach 3: Most Efficient Solution
        def most_efficient_solution(nums: List[int]) -> int:
            """
            Most efficient approach - often involves mathematical insights
            """
            # Mathematical approach or advanced algorithm
            return result
        
        # Return the most efficient solution
        return optimized_solution(nums)


# Common Array & Hashing Patterns

class ArrayPatterns:
    """Common patterns and techniques for array problems"""
    
    @staticmethod
    def two_sum(nums: List[int], target: int) -> List[int]:
        """
        Two Sum - Classic hash map approach
        Time: O(n), Space: O(n)
        """
        seen = {}  # val -> index
        
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        
        return []
    
    @staticmethod
    def contains_duplicate(nums: List[int]) -> bool:
        """
        Contains Duplicate - Hash set approach
        Time: O(n), Space: O(n)
        """
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False
    
    @staticmethod
    def valid_anagram(s: str, t: str) -> bool:
        """
        Valid Anagram - Counter approach
        Time: O(n), Space: O(1) since fixed character set
        """
        return Counter(s) == Counter(t)
    
    @staticmethod
    def group_anagrams(strs: List[str]) -> List[List[str]]:
        """
        Group Anagrams - Hash map with sorted key
        Time: O(n * k * log k), Space: O(n * k)
        where n = number of strings, k = max string length
        """
        groups = defaultdict(list)
        
        for s in strs:
            # Sort characters to create key
            key = ''.join(sorted(s))
            groups[key].append(s)
        
        return list(groups.values())
    
    @staticmethod
    def top_k_frequent(nums: List[int], k: int) -> List[int]:
        """
        Top K Frequent Elements - Counter + heap approach
        Time: O(n + k log n), Space: O(n)
        """
        from heapq import nlargest
        
        # Count frequencies
        count = Counter(nums)
        
        # Get top k elements
        return nlargest(k, count.keys(), key=count.get)
    
    @staticmethod
    def product_except_self(nums: List[int]) -> List[int]:
        """
        Product of Array Except Self - Prefix and suffix approach
        Time: O(n), Space: O(1) excluding output array
        """
        n = len(nums)
        result = [1] * n
        
        # Calculate prefix products
        prefix = 1
        for i in range(n):
            result[i] = prefix
            prefix *= nums[i]
        
        # Calculate suffix products and combine
        suffix = 1
        for i in range(n - 1, -1, -1):
            result[i] *= suffix
            suffix *= nums[i]
        
        return result


# Testing and Benchmarking

def test_solutions():
    """Test function to verify solutions work correctly"""
    
    # Test cases
    test_cases = [
        {
            "name": "Two Sum",
            "input": ([2, 7, 11, 15], 9),
            "expected": [0, 1]
        },
        {
            "name": "Contains Duplicate",
            "input": ([1, 2, 3, 1],),
            "expected": True
        },
        {
            "name": "Valid Anagram",
            "input": ("anagram", "nagaram"),
            "expected": True
        }
    ]
    
    patterns = ArrayPatterns()
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        
        if test['name'] == "Two Sum":
            result = patterns.two_sum(*test['input'])
        elif test['name'] == "Contains Duplicate":
            result = patterns.contains_duplicate(*test['input'])
        elif test['name'] == "Valid Anagram":
            result = patterns.valid_anagram(*test['input'])
        
        print(f"Input: {test['input']}")
        print(f"Expected: {test['expected']}")
        print(f"Result: {result}")
        print(f"Pass: {result == test['expected']}")


def benchmark_solutions():
    """Benchmark different approaches to compare performance"""
    
    # Example: Compare different approaches for the same problem
    nums = list(range(10000))
    target = 19998
    
    print("Benchmarking Two Sum approaches:")
    
    # Approach 1: Brute Force
    start_time = time.time()
    # brute_force_result = brute_force_two_sum(nums, target)
    brute_force_time = time.time() - start_time
    
    # Approach 2: Hash Map
    start_time = time.time()
    hash_map_result = ArrayPatterns.two_sum(nums, target)
    hash_map_time = time.time() - start_time
    
    print(f"Brute Force: {brute_force_time:.6f} seconds")
    print(f"Hash Map: {hash_map_time:.6f} seconds")
    print(f"Speedup: {brute_force_time / hash_map_time:.2f}x")


if __name__ == "__main__":
    print("Arrays & Hashing Solutions - NeetCode Style")
    print("=" * 50)
    
    # Run tests
    test_solutions()
    
    # Run benchmarks
    print("\n" + "=" * 50)
    benchmark_solutions()
    
    print("\nTemplate ready for use! Add your specific problem solutions here.")
