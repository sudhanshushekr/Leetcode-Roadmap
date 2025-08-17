"""
Sliding Window Problems - NeetCode Style Template

This template provides a structured approach to solving sliding window problems
with multiple solutions, complexity analysis, and clear explanations.

Common Patterns:
1. Fixed Size Window
2. Variable Size Window
3. Two Pointers with Window
4. Prefix Sum with Window
5. Monotonic Queue/Stack
6. Sliding Window with Hash Map
"""

from typing import List, Dict, Set
from collections import defaultdict, deque
import time


class Solution:
    """
    Main solution class following NeetCode's structure
    """
    
    def problem_name(self, nums: List[int], k: int) -> int:
        """
        Problem: [Brief description of the problem]
        
        Example:
        Input: nums = [1, 2, 3, 4, 5], k = 3
        Output: 12
        
        Approach 1: Brute Force
        Time Complexity: O(n * k)
        Space Complexity: O(1)
        """
        
        # Approach 1: Brute Force Solution
        def brute_force_solution(nums: List[int], k: int) -> int:
            """
            Brute force approach - check all possible windows
            """
            result = 0
            for i in range(len(nums) - k + 1):
                window_sum = sum(nums[i:i + k])
                result = max(result, window_sum)
            return result
        
        # Approach 2: Sliding Window Solution
        def sliding_window_solution(nums: List[int], k: int) -> int:
            """
            Sliding window approach - efficient solution
            """
            if not nums or k <= 0:
                return 0
            
            # Initialize first window
            window_sum = sum(nums[:k])
            result = window_sum
            
            # Slide the window
            for i in range(k, len(nums)):
                window_sum = window_sum - nums[i - k] + nums[i]
                result = max(result, window_sum)
            
            return result
        
        # Approach 3: Most Efficient Solution
        def most_efficient_solution(nums: List[int], k: int) -> int:
            """
            Most efficient approach - often involves mathematical insights
            """
            # Mathematical approach or advanced algorithm
            return result
        
        # Return the most efficient solution
        return sliding_window_solution(nums, k)


# Common Sliding Window Patterns

class SlidingWindowPatterns:
    """Common patterns and techniques for sliding window problems"""
    
    @staticmethod
    def max_sum_subarray_fixed_k(nums: List[int], k: int) -> int:
        """
        Maximum Sum Subarray of Size K - Fixed window
        Time: O(n), Space: O(1)
        """
        if not nums or k <= 0 or k > len(nums):
            return 0
        
        # Calculate first window
        window_sum = sum(nums[:k])
        max_sum = window_sum
        
        # Slide window
        for i in range(k, len(nums)):
            window_sum = window_sum - nums[i - k] + nums[i]
            max_sum = max(max_sum, window_sum)
        
        return max_sum
    
    @staticmethod
    def min_subarray_sum_target(nums: List[int], target: int) -> int:
        """
        Minimum Size Subarray Sum - Variable window
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        
        left, window_sum = 0, 0
        min_length = float('inf')
        
        for right in range(len(nums)):
            window_sum += nums[right]
            
            while window_sum >= target:
                min_length = min(min_length, right - left + 1)
                window_sum -= nums[left]
                left += 1
        
        return min_length if min_length != float('inf') else 0
    
    @staticmethod
    def longest_substring_k_distinct(s: str, k: int) -> int:
        """
        Longest Substring with At Most K Distinct Characters
        Time: O(n), Space: O(k)
        """
        if not s or k <= 0:
            return 0
        
        char_count = defaultdict(int)
        left, max_length = 0, 0
        
        for right in range(len(s)):
            char_count[s[right]] += 1
            
            while len(char_count) > k:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    @staticmethod
    def longest_substring_no_repeating(s: str) -> int:
        """
        Longest Substring Without Repeating Characters
        Time: O(n), Space: O(min(m, n)) where m is charset size
        """
        if not s:
            return 0
        
        char_index = {}
        left, max_length = 0, 0
        
        for right in range(len(s)):
            if s[right] in char_index and char_index[s[right]] >= left:
                left = char_index[s[right]] + 1
            
            char_index[s[right]] = right
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    @staticmethod
    def max_sliding_window(nums: List[int], k: int) -> List[int]:
        """
        Sliding Window Maximum - Using monotonic queue
        Time: O(n), Space: O(k)
        """
        if not nums or k <= 0:
            return []
        
        result = []
        dq = deque()  # Store indices
        
        for i in range(len(nums)):
            # Remove elements outside window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove smaller elements from back
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            dq.append(i)
            
            # Add maximum to result
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    @staticmethod
    def find_anagrams(s: str, p: str) -> List[int]:
        """
        Find All Anagrams in a String
        Time: O(n), Space: O(1) - fixed character set
        """
        if not s or not p or len(p) > len(s):
            return []
        
        p_count = defaultdict(int)
        s_count = defaultdict(int)
        
        # Count characters in pattern
        for char in p:
            p_count[char] += 1
        
        result = []
        left = 0
        
        for right in range(len(s)):
            s_count[s[right]] += 1
            
            # Shrink window if it exceeds pattern length
            if right - left + 1 > len(p):
                s_count[s[left]] -= 1
                if s_count[s[left]] == 0:
                    del s_count[s[left]]
                left += 1
            
            # Check if current window is an anagram
            if s_count == p_count:
                result.append(left)
        
        return result


# Testing and Benchmarking

def test_solutions():
    """Test function to verify solutions work correctly"""
    
    # Test cases
    test_cases = [
        {
            "name": "Max Sum Subarray Fixed K",
            "input": ([1, 4, 2, 10, 2, 3, 1, 0, 20], 4),
            "expected": 24
        },
        {
            "name": "Min Subarray Sum Target",
            "input": ([2, 3, 1, 2, 4, 3], 7),
            "expected": 2
        },
        {
            "name": "Longest Substring K Distinct",
            "input": ("eceba", 2),
            "expected": 3
        }
    ]
    
    patterns = SlidingWindowPatterns()
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        
        if test['name'] == "Max Sum Subarray Fixed K":
            result = patterns.max_sum_subarray_fixed_k(*test['input'])
        elif test['name'] == "Min Subarray Sum Target":
            result = patterns.min_subarray_sum_target(*test['input'])
        elif test['name'] == "Longest Substring K Distinct":
            result = patterns.longest_substring_k_distinct(*test['input'])
        
        print(f"Input: {test['input']}")
        print(f"Expected: {test['expected']}")
        print(f"Result: {result}")
        print(f"Pass: {result == test['expected']}")


def benchmark_solutions():
    """Benchmark different approaches to compare performance"""
    
    # Example: Compare different approaches for the same problem
    nums = list(range(10000))
    k = 100
    
    print("Benchmarking Sliding Window approaches:")
    
    # Approach 1: Brute Force
    start_time = time.time()
    # brute_force_result = brute_force_max_sum(nums, k)
    brute_force_time = time.time() - start_time
    
    # Approach 2: Sliding Window
    start_time = time.time()
    sliding_window_result = SlidingWindowPatterns.max_sum_subarray_fixed_k(nums, k)
    sliding_window_time = time.time() - start_time
    
    print(f"Brute Force: {brute_force_time:.6f} seconds")
    print(f"Sliding Window: {sliding_window_time:.6f} seconds")
    print(f"Speedup: {brute_force_time / sliding_window_time:.2f}x")


if __name__ == "__main__":
    print("Sliding Window Solutions - NeetCode Style")
    print("=" * 50)
    
    # Run tests
    test_solutions()
    
    # Run benchmarks
    print("\n" + "=" * 50)
    benchmark_solutions()
    
    print("\nTemplate ready for use! Add your specific problem solutions here.")
