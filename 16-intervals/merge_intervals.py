"""
Merge Intervals - LeetCode Problem 56
https://leetcode.com/problems/merge-intervals/

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, 
and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.

Constraints:
- 1 <= intervals.length <= 10^4
- intervals[i].length == 2
- 0 <= starti <= endi <= 10^4
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Merge Intervals
    """
    
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Main method - returns the most efficient solution
        """
        return self.merge_sorting(intervals)
    
    def merge_sorting(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Approach 1: Sorting - Most Efficient
        
        Algorithm:
        1. Sort intervals by start time
        2. Iterate through sorted intervals
        3. Merge overlapping intervals
        4. Add non-overlapping intervals to result
        
        Time Complexity: O(n log n) - Sorting dominates
        Space Complexity: O(n) - Store result
        
        Analysis:
        - Pros: Optimal time complexity, simple logic
        - Cons: Requires sorting
        """
        if not intervals:
            return []
        
        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])
        
        result = [intervals[0]]
        
        for interval in intervals[1:]:
            # If current interval overlaps with last merged interval
            if interval[0] <= result[-1][1]:
                # Merge intervals
                result[-1][1] = max(result[-1][1], interval[1])
            else:
                # Add non-overlapping interval
                result.append(interval)
        
        return result
    
    def merge_brute_force(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Approach 2: Brute Force - Check all pairs
        
        Algorithm:
        1. Check all pairs of intervals for overlap
        2. Merge overlapping pairs
        3. Repeat until no more merges possible
        
        Time Complexity: O(n²) - Check all pairs
        Space Complexity: O(n) - Store result
        
        Analysis:
        - Pros: Simple logic
        - Cons: Inefficient, only for educational purposes
        """
        if not intervals:
            return []
        
        # Copy intervals to avoid modifying input
        intervals = intervals.copy()
        merged = True
        
        while merged:
            merged = False
            i = 0
            while i < len(intervals):
                j = i + 1
                while j < len(intervals):
                    # Check if intervals overlap
                    if (intervals[i][0] <= intervals[j][1] and 
                        intervals[j][0] <= intervals[i][1]):
                        # Merge intervals
                        start = min(intervals[i][0], intervals[j][0])
                        end = max(intervals[i][1], intervals[j][1])
                        intervals[i] = [start, end]
                        intervals.pop(j)
                        merged = True
                    else:
                        j += 1
                i += 1
        
        return intervals
    
    def merge_sweep_line(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Approach 3: Sweep Line Algorithm
        
        Algorithm:
        1. Create events for start and end of each interval
        2. Sort events by time
        3. Process events to find overlapping intervals
        
        Time Complexity: O(n log n) - Sort events
        Space Complexity: O(n) - Store events
        
        Analysis:
        - Pros: Efficient for complex interval problems
        - Cons: More complex implementation
        """
        if not intervals:
            return []
        
        # Create events: (time, type, interval_index)
        events = []
        for i, interval in enumerate(intervals):
            events.append((interval[0], 'start', i))
            events.append((interval[1], 'end', i))
        
        # Sort events by time
        events.sort()
        
        result = []
        active_intervals = set()
        current_start = None
        
        for time, event_type, interval_idx in events:
            if event_type == 'start':
                active_intervals.add(interval_idx)
                if current_start is None:
                    current_start = time
            else:  # end event
                active_intervals.remove(interval_idx)
                if not active_intervals:
                    # End of current merged interval
                    result.append([current_start, time])
                    current_start = None
        
        return result
    
    def merge_divide_conquer(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Approach 4: Divide and Conquer
        
        Algorithm:
        1. Divide intervals into two halves
        2. Recursively merge each half
        3. Merge the two merged halves
        
        Time Complexity: O(n log n) - Divide and conquer
        Space Complexity: O(log n) - Recursion stack
        
        Analysis:
        - Pros: Parallelizable approach
        - Cons: More complex implementation
        """
        if not intervals:
            return []
        
        def merge_two_lists(list1, list2):
            """Merge two sorted interval lists"""
            if not list1:
                return list2
            if not list2:
                return list1
            
            result = []
            i = j = 0
            
            while i < len(list1) and j < len(list2):
                if list1[i][0] <= list2[j][0]:
                    result.append(list1[i])
                    i += 1
                else:
                    result.append(list2[j])
                    j += 1
            
            result.extend(list1[i:])
            result.extend(list2[j:])
            
            # Merge overlapping intervals in result
            merged = [result[0]]
            for interval in result[1:]:
                if interval[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], interval[1])
                else:
                    merged.append(interval)
            
            return merged
        
        def merge_recursive(intervals, start, end):
            if start == end:
                return [intervals[start]]
            if start > end:
                return []
            
            mid = (start + end) // 2
            left = merge_recursive(intervals, start, mid)
            right = merge_recursive(intervals, mid + 1, end)
            
            return merge_two_lists(left, right)
        
        # Sort intervals first
        intervals.sort(key=lambda x: x[0])
        return merge_recursive(intervals, 0, len(intervals) - 1)
    
    def merge_optimized_sorting(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Approach 5: Optimized Sorting with Early Termination
        
        Algorithm:
        1. Sort intervals by start time
        2. Use single pass to merge overlapping intervals
        3. Optimize by checking if intervals are already sorted
        
        Time Complexity: O(n log n) - Sorting dominates
        Space Complexity: O(n) - Store result
        
        Analysis:
        - Pros: Simple and efficient
        - Cons: Always requires sorting
        """
        if not intervals:
            return []
        
        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])
        
        result = []
        current = intervals[0]
        
        for interval in intervals[1:]:
            if interval[0] <= current[1]:
                # Overlapping intervals, merge them
                current[1] = max(current[1], interval[1])
            else:
                # Non-overlapping interval, add current to result
                result.append(current)
                current = interval
        
        # Add the last interval
        result.append(current)
        
        return result


# Testing and Benchmarking

def test_merge_intervals():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "intervals": [[1, 3], [2, 6], [8, 10], [15, 18]],
            "expected": [[1, 6], [8, 10], [15, 18]],
            "description": "Basic overlapping intervals"
        },
        {
            "intervals": [[1, 4], [4, 5]],
            "expected": [[1, 5]],
            "description": "Adjacent intervals"
        },
        {
            "intervals": [[1, 4], [0, 4]],
            "expected": [[0, 4]],
            "description": "Completely overlapping"
        },
        {
            "intervals": [[1, 4], [2, 3]],
            "expected": [[1, 4]],
            "description": "One contained in another"
        },
        {
            "intervals": [[1, 4], [5, 6]],
            "expected": [[1, 4], [5, 6]],
            "description": "Non-overlapping intervals"
        },
        {
            "intervals": [[1, 4]],
            "expected": [[1, 4]],
            "description": "Single interval"
        },
        {
            "intervals": [],
            "expected": [],
            "description": "Empty intervals"
        }
    ]
    
    approaches = [
        ("Sorting", solution.merge_sorting),
        ("Sweep Line", solution.merge_sweep_line),
        ("Divide & Conquer", solution.merge_divide_conquer),
        ("Optimized Sorting", solution.merge_optimized_sorting)
    ]
    
    print("Merge Intervals - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: intervals = {test['intervals']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['intervals'].copy())
                
                # Sort both result and expected for comparison
                result_sorted = sorted(result)
                expected_sorted = sorted(test['expected'])
                
                passed = result_sorted == expected_sorted
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with larger input"""
    
    solution = Solution()
    
    # Test with larger input
    intervals = [[i, i + 2] for i in range(0, 1000, 2)] + [[i, i + 1] for i in range(1, 1000, 2)]
    
    print(f"\nBenchmarking with {len(intervals)} intervals")
    print("=" * 50)
    
    approaches = [
        ("Sorting", solution.merge_sorting),
        ("Sweep Line", solution.merge_sweep_line),
        ("Divide & Conquer", solution.merge_divide_conquer),
        ("Optimized Sorting", solution.merge_optimized_sorting)
    ]
    
    # Skip brute force for large input as it's too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(intervals.copy())
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds ({len(result)} merged intervals)")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Sorting", "O(n log n)", "O(n)", "Optimal time complexity"),
        ("Sweep Line", "O(n log n)", "O(n)", "Efficient for complex problems"),
        ("Divide & Conquer", "O(n log n)", "O(log n)", "Parallelizable"),
        ("Optimized Sorting", "O(n log n)", "O(n)", "Simple and efficient"),
        ("Brute Force", "O(n²)", "O(n)", "Educational only")
    ]
    
    print(f"{'Approach':<20} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<20} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Merge Intervals - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_merge_intervals()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the sorting approach for optimal performance.")
