# Two Pointers - LeetCode Solutions

This folder contains solutions for two pointers problems following NeetCode's structured approach.

## 📁 File Structure

- `two_pointers_template.py` - Template for creating new solutions
- `two_sum_ii.py` - Complete solution for Two Sum II (LeetCode #167)
- `README.md` - This file with patterns and guidelines

## 🎯 Common Patterns

### 1. Two Pointers from Ends
- **Use when**: Array is sorted, need to find pairs, or working from both ends
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(1) usually
- **Examples**: Two Sum II, Container With Most Water, Valid Palindrome

### 2. Fast and Slow Pointers
- **Use when**: Need to detect cycles, find middle element, or work with linked lists
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(1) usually
- **Examples**: Linked List Cycle, Middle of Linked List, Remove Nth Node

### 3. Sliding Window with Two Pointers
- **Use when**: Need to find subarrays with certain properties
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(1) or O(k) where k is window size
- **Examples**: Longest Substring Without Repeating Characters

### 4. Three Pointers
- **Use when**: Need to work with three elements or arrays
- **Time Complexity**: O(n²) typically
- **Space Complexity**: O(1) usually
- **Examples**: 3Sum, 4Sum, Merge Three Sorted Arrays

### 5. Merge with Two Pointers
- **Use when**: Need to merge sorted arrays or lists
- **Time Complexity**: O(m + n) typically
- **Space Complexity**: O(1) or O(m + n)
- **Examples**: Merge Sorted Array, Merge Two Sorted Lists

### 6. Partition with Two Pointers
- **Use when**: Need to partition array based on some condition
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(1) usually
- **Examples**: Sort Colors, Move Zeroes, Partition List

## 🚀 Solution Template Structure

Each solution follows this structure:

```python
class Solution:
    def problem_name(self, nums: List[int]) -> int:
        """
        Problem: [Brief description]
        
        Example:
        Input: nums = [1, 2, 3, 4, 5]
        Output: 15
        
        Approach 1: Brute Force
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        
        # Approach 1: Brute Force
        def brute_force_solution(nums):
            # Implementation
            pass
        
        # Approach 2: Two Pointers
        def two_pointers_solution(nums):
            # Implementation
            pass
        
        # Return best solution
        return two_pointers_solution(nums)
```

## 📊 Complexity Analysis

| Pattern | Time Complexity | Space Complexity | When to Use |
|---------|----------------|------------------|-------------|
| Two Pointers from Ends | O(n) | O(1) | Sorted arrays, pairs |
| Fast and Slow Pointers | O(n) | O(1) | Cycles, middle element |
| Sliding Window | O(n) | O(1) | Subarray problems |
| Three Pointers | O(n²) | O(1) | Three element problems |
| Merge with Two Pointers | O(m+n) | O(1) | Merging sorted data |
| Partition | O(n) | O(1) | Array partitioning |

## 🧪 Testing Strategy

Each solution includes:

1. **Multiple Test Cases**: Edge cases, normal cases, large inputs
2. **Benchmarking**: Compare different approaches
3. **Complexity Analysis**: Time and space complexity for each approach
4. **Error Handling**: Graceful handling of edge cases

## 📝 Common Problems in This Category

### Easy Problems
- [ ] Two Sum II (LeetCode #167)
- [ ] Valid Palindrome (LeetCode #125)
- [ ] Remove Duplicates from Sorted Array (LeetCode #26)
- [ ] Move Zeroes (LeetCode #283)

### Medium Problems
- [ ] Container With Most Water (LeetCode #11)
- [ ] 3Sum (LeetCode #15)
- [ ] Sort Colors (LeetCode #75)
- [ ] Remove Nth Node From End of List (LeetCode #19)

### Hard Problems
- [ ] Trapping Rain Water (LeetCode #42)
- [ ] 4Sum (LeetCode #18)
- [ ] Median of Two Sorted Arrays (LeetCode #4)

## 🎯 Key Insights

1. **Two pointers are versatile**: Can solve many array and linked list problems efficiently
2. **Consider array properties**: Sorted arrays enable more efficient two pointer solutions
3. **Fast and slow pointers**: Very useful for linked list problems
4. **Sliding window**: Often implemented with two pointers
5. **Edge cases**: Always consider empty arrays, single elements, duplicates

## 🔧 How to Use This Template

1. **Copy the template**: Use `two_pointers_template.py` as a starting point
2. **Implement multiple approaches**: Start with brute force, then optimize
3. **Add test cases**: Include edge cases and large inputs
4. **Benchmark solutions**: Compare performance of different approaches
5. **Document complexity**: Always note time and space complexity

## 📚 Additional Resources

- [NeetCode Two Pointers Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53ldVwtstag2TL4HQhAnC8ATf)
- [LeetCode Two Pointers Tag](https://leetcode.com/tag/two-pointers/)

## 🎉 Tips for Success

1. **Start with brute force**: Always implement the obvious solution first
2. **Look for sorted arrays**: They often enable efficient two pointer solutions
3. **Use fast and slow pointers**: Great for linked list problems
4. **Consider sliding window**: Many problems can be solved with two pointers
5. **Practice regularly**: Two pointers are fundamental concepts

---

**Happy Coding! 🚀**
