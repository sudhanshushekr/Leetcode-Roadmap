"""
Two Pointers Problems - NeetCode Style Template

This template provides a structured approach to solving two pointers problems
with multiple solutions, complexity analysis, and clear explanations.

Common Patterns:
1. Two Pointers from Ends
2. Fast and Slow Pointers
3. Sliding Window with Two Pointers
4. Three Pointers
5. Merge with Two Pointers
6. Partition with Two Pointers
"""

from typing import List, Optional
import time


class Solution:
    """
    Main solution class following NeetCode's structure
    """
    
    def problem_name(self, nums: List[int]) -> int:
        """
        Problem: [Brief description of the problem]
        
        Example:
        Input: nums = [1, 2, 3, 4, 5]
        Output: 15
        
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
        
        # Approach 2: Two Pointers Solution
        def two_pointers_solution(nums: List[int]) -> int:
            """
            Two pointers approach - efficient solution
            """
            left, right = 0, len(nums) - 1
            result = 0
            
            while left < right:
                # Your two pointers logic here
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
        return two_pointers_solution(nums)


# Common Two Pointers Patterns

class TwoPointersPatterns:
    """Common patterns and techniques for two pointers problems"""
    
    @staticmethod
    def two_sum_sorted(nums: List[int], target: int) -> List[int]:
        """
        Two Sum II - Two pointers approach for sorted array
        Time: O(n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                return [left + 1, right + 1]  # 1-indexed
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []
    
    @staticmethod
    def remove_duplicates_sorted(nums: List[int]) -> int:
        """
        Remove Duplicates from Sorted Array - Two pointers
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        
        write_index = 1
        
        for read_index in range(1, len(nums)):
            if nums[read_index] != nums[read_index - 1]:
                nums[write_index] = nums[read_index]
                write_index += 1
        
        return write_index
    
    @staticmethod
    def container_with_most_water(height: List[int]) -> int:
        """
        Container With Most Water - Two pointers from ends
        Time: O(n), Space: O(1)
        """
        left, right = 0, len(height) - 1
        max_area = 0
        
        while left < right:
            width = right - left
            h = min(height[left], height[right])
            area = width * h
            max_area = max(max_area, area)
            
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
    
    @staticmethod
    def three_sum(nums: List[int]) -> List[List[int]]:
        """
        3Sum - Three pointers approach
        Time: O(n²), Space: O(1) excluding output
        """
        nums.sort()
        result = []
        n = len(nums)
        
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            left, right = i + 1, n - 1
            
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                
                if total == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # Skip duplicates
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        
        return result
    
    @staticmethod
    def merge_sorted_arrays(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Merge Sorted Array - Two pointers from end
        Time: O(m + n), Space: O(1)
        """
        p1, p2, p = m - 1, n - 1, m + n - 1
        
        while p2 >= 0:
            if p1 >= 0 and nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
    
    @staticmethod
    def linked_list_cycle(head: Optional['ListNode']) -> bool:
        """
        Linked List Cycle - Fast and slow pointers
        Time: O(n), Space: O(1)
        """
        if not head or not head.next:
            return False
        
        slow = head
        fast = head.next
        
        while slow != fast:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        
        return True


# Testing and Benchmarking

def test_solutions():
    """Test function to verify solutions work correctly"""
    
    # Test cases
    test_cases = [
        {
            "name": "Two Sum Sorted",
            "input": ([2, 7, 11, 15], 9),
            "expected": [1, 2]
        },
        {
            "name": "Remove Duplicates",
            "input": ([1, 1, 2, 2, 3],),
            "expected": 3
        },
        {
            "name": "Container With Most Water",
            "input": ([1, 8, 6, 2, 5, 4, 8, 3, 7],),
            "expected": 49
        }
    ]
    
    patterns = TwoPointersPatterns()
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        
        if test['name'] == "Two Sum Sorted":
            result = patterns.two_sum_sorted(*test['input'])
        elif test['name'] == "Remove Duplicates":
            nums = list(test['input'][0])  # Copy to avoid modifying original
            result = patterns.remove_duplicates_sorted(nums)
        elif test['name'] == "Container With Most Water":
            result = patterns.container_with_most_water(*test['input'])
        
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
    
    # Approach 2: Two Pointers
    start_time = time.time()
    two_pointers_result = TwoPointersPatterns.two_sum_sorted(nums, target)
    two_pointers_time = time.time() - start_time
    
    print(f"Brute Force: {brute_force_time:.6f} seconds")
    print(f"Two Pointers: {two_pointers_time:.6f} seconds")
    print(f"Speedup: {brute_force_time / two_pointers_time:.2f}x")


if __name__ == "__main__":
    print("Two Pointers Solutions - NeetCode Style")
    print("=" * 50)
    
    # Run tests
    test_solutions()
    
    # Run benchmarks
    print("\n" + "=" * 50)
    benchmark_solutions()
    
    print("\nTemplate ready for use! Add your specific problem solutions here.")
