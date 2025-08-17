"""
Top K Frequent Elements - LeetCode Problem 347
https://leetcode.com/problems/top-k-frequent-elements/

Given an integer array nums and an integer k, return the k most frequent elements. 
You may return the answer in any order.

Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]

Constraints:
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
- k is in the range [1, the number of unique elements in the array].
- It is guaranteed that the answer is unique.
"""

from typing import List
import time
from collections import Counter, defaultdict
import heapq


class Solution:
    """
    Solution class with multiple approaches to solve Top K Frequent Elements
    """
    
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        Main method - returns the most efficient solution
        """
        return self.top_k_frequent_heap(nums, k)
    
    def top_k_frequent_heap(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 1: Heap (Priority Queue) - Most Efficient
        
        Algorithm:
        1. Count frequency of each element using Counter
        2. Use min-heap to keep top k frequent elements
        3. Return the k elements from heap
        
        Time Complexity: O(n + k log n) - Count + heap operations
        Space Complexity: O(n) - Store frequency count and heap
        
        Analysis:
        - Pros: Optimal for large k, efficient heap operations
        - Cons: Slightly more complex implementation
        """
        # Count frequencies
        count = Counter(nums)
        
        # Use min-heap to keep top k frequent elements
        # Store (-frequency, element) to simulate max-heap with min-heap
        heap = []
        for num, freq in count.items():
            heapq.heappush(heap, (-freq, num))
        
        # Extract top k elements
        result = []
        for _ in range(k):
            result.append(heapq.heappop(heap)[1])
        
        return result
    
    def top_k_frequent_nlargest(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 2: Using heapq.nlargest - Clean Implementation
        
        Algorithm:
        1. Count frequency of each element
        2. Use heapq.nlargest to get top k elements by frequency
        
        Time Complexity: O(n + k log n) - Count + nlargest operation
        Space Complexity: O(n) - Store frequency count
        
        Analysis:
        - Pros: Very clean implementation, built-in optimization
        - Cons: Same complexity as manual heap approach
        """
        count = Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)
    
    def top_k_frequent_sorting(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 3: Sorting - Simple but Less Efficient
        
        Algorithm:
        1. Count frequency of each element
        2. Sort by frequency in descending order
        3. Return first k elements
        
        Time Complexity: O(n + m log m) - Count + sort unique elements
        Space Complexity: O(n) - Store frequency count
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Less efficient than heap approach
        """
        count = Counter(nums)
        
        # Sort by frequency (descending) and return first k elements
        sorted_items = sorted(count.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:k]]
    
    def top_k_frequent_bucket_sort(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 4: Bucket Sort - Linear Time for Small k
        
        Algorithm:
        1. Count frequency of each element
        2. Create buckets where bucket[i] contains elements with frequency i
        3. Iterate from highest frequency bucket to lowest
        4. Collect k elements
        
        Time Complexity: O(n) - Linear time
        Space Complexity: O(n) - Store frequency count and buckets
        
        Analysis:
        - Pros: Linear time complexity, very efficient
        - Cons: More complex implementation, best when k is small
        """
        count = Counter(nums)
        
        # Create buckets: bucket[i] contains elements with frequency i
        buckets = defaultdict(list)
        max_freq = 0
        
        for num, freq in count.items():
            buckets[freq].append(num)
            max_freq = max(max_freq, freq)
        
        # Collect top k elements from highest frequency buckets
        result = []
        for freq in range(max_freq, 0, -1):
            if freq in buckets:
                result.extend(buckets[freq])
                if len(result) >= k:
                    break
        
        return result[:k]
    
    def top_k_frequent_quickselect(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 5: Quickselect - Average Linear Time
        
        Algorithm:
        1. Count frequency of each element
        2. Use quickselect to find kth most frequent element
        3. Return all elements with frequency >= kth element
        
        Time Complexity: O(n) average case, O(n²) worst case
        Space Complexity: O(n) - Store frequency count
        
        Analysis:
        - Pros: Average linear time, in-place partitioning
        - Cons: Worst case quadratic, more complex
        """
        count = Counter(nums)
        unique_nums = list(count.keys())
        
        def partition(left, right, pivot_index):
            pivot_freq = count[unique_nums[pivot_index]]
            
            # Move pivot to end
            unique_nums[pivot_index], unique_nums[right] = unique_nums[right], unique_nums[pivot_index]
            
            # Move all less frequent elements to the left
            store_index = left
            for i in range(left, right):
                if count[unique_nums[i]] < pivot_freq:
                    unique_nums[store_index], unique_nums[i] = unique_nums[i], unique_nums[store_index]
                    store_index += 1
            
            # Move pivot to its final place
            unique_nums[right], unique_nums[store_index] = unique_nums[store_index], unique_nums[right]
            
            return store_index
        
        def quickselect(left, right, k_smallest):
            if left == right:
                return
            
            # Select a random pivot
            import random
            pivot_index = random.randint(left, right)
            
            # Find the pivot position in a sorted array
            pivot_index = partition(left, right, pivot_index)
            
            if k_smallest == pivot_index:
                return
            elif k_smallest < pivot_index:
                quickselect(left, pivot_index - 1, k_smallest)
            else:
                quickselect(pivot_index + 1, right, k_smallest)
        
        # kth top frequent element is (n - k)th less frequent
        n = len(unique_nums)
        quickselect(0, n - 1, n - k)
        
        # Return top k frequent elements
        return unique_nums[n - k:]


# Testing and Benchmarking

def test_top_k_frequent():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "nums": [1, 1, 1, 2, 2, 3],
            "k": 2,
            "expected": [1, 2],
            "description": "Basic case"
        },
        {
            "nums": [1],
            "k": 1,
            "expected": [1],
            "description": "Single element"
        },
        {
            "nums": [1, 2, 3, 4, 5],
            "k": 3,
            "expected": [1, 2, 3],
            "description": "All elements have same frequency"
        },
        {
            "nums": [1, 1, 1, 2, 2, 2, 3, 3, 4],
            "k": 2,
            "expected": [1, 2],
            "description": "Tie for top frequency"
        },
        {
            "nums": [-1, -1, 1, 1, 1, 2, 2, 3],
            "k": 3,
            "expected": [1, -1, 2],
            "description": "Negative numbers"
        }
    ]
    
    approaches = [
        ("Heap", solution.top_k_frequent_heap),
        ("Nlargest", solution.top_k_frequent_nlargest),
        ("Sorting", solution.top_k_frequent_sorting),
        ("Bucket Sort", solution.top_k_frequent_bucket_sort),
        ("Quickselect", solution.top_k_frequent_quickselect)
    ]
    
    print("Top K Frequent Elements - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: nums = {test['nums']}, k = {test['k']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['nums'], test['k'])
                
                # Sort both result and expected for comparison (order doesn't matter)
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
    nums = []
    for i in range(n):
        # Create some elements with high frequency
        if i % 3 == 0:
            nums.extend([1] * 10)  # High frequency
        elif i % 3 == 1:
            nums.extend([2] * 5)   # Medium frequency
        else:
            nums.append(i)         # Low frequency
    
    k = 5
    
    print(f"\nBenchmarking with array of size {len(nums)}, k = {k}")
    print("=" * 50)
    
    approaches = [
        ("Heap", solution.top_k_frequent_heap),
        ("Nlargest", solution.top_k_frequent_nlargest),
        ("Sorting", solution.top_k_frequent_sorting),
        ("Bucket Sort", solution.top_k_frequent_bucket_sort)
    ]
    
    # Skip quickselect for large input as it can be slow in worst case
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(nums, k)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds")
        print(f"  Result: {result}")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Heap", "O(n + k log n)", "O(n)", "Optimal for large k"),
        ("Nlargest", "O(n + k log n)", "O(n)", "Clean implementation"),
        ("Sorting", "O(n + m log m)", "O(n)", "Simple but less efficient"),
        ("Bucket Sort", "O(n)", "O(n)", "Linear time, best for small k"),
        ("Quickselect", "O(n) avg", "O(n)", "Average linear time")
    ]
    
    print(f"{'Approach':<15} {'Time':<15} {'Space':<10} {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<15} {time_comp:<15} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Top K Frequent Elements - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_top_k_frequent()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use heap approach for optimal performance.")
