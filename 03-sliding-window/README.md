# Sliding Window - LeetCode Solutions

This folder contains solutions for sliding window problems following NeetCode's structured approach.

## 📁 File Structure

- `sliding_window_template.py` - Template for creating new solutions
- `README.md` - This file with patterns and guidelines

## 🎯 Common Patterns

### 1. Fixed Size Window
- **Use when**: Need to find maximum/minimum of subarrays of fixed size
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(1) usually
- **Examples**: Maximum Sum Subarray of Size K, Sliding Window Maximum

### 2. Variable Size Window
- **Use when**: Need to find smallest/largest subarray satisfying a condition
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(1) or O(k) where k is window size
- **Examples**: Minimum Size Subarray Sum, Longest Substring with K Distinct Characters

### 3. Two Pointers with Window
- **Use when**: Need to maintain a window with two pointers
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(1) usually
- **Examples**: Longest Substring Without Repeating Characters

### 4. Prefix Sum with Window
- **Use when**: Need to calculate sum of subarrays efficiently
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(n) for prefix array
- **Examples**: Subarray Sum Equals K

### 5. Monotonic Queue/Stack
- **Use when**: Need to maintain max/min in sliding window
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(k) where k is window size
- **Examples**: Sliding Window Maximum, Sliding Window Minimum

### 6. Sliding Window with Hash Map
- **Use when**: Need to track character frequencies or unique elements
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(k) where k is unique elements
- **Examples**: Longest Substring with At Most K Distinct Characters

## 🚀 Solution Template Structure

Each solution follows this structure:

```python
class Solution:
    def problem_name(self, nums: List[int], k: int) -> int:
        """
        Problem: [Brief description]
        
        Example:
        Input: nums = [1, 2, 3, 4, 5], k = 3
        Output: 12
        
        Approach 1: Brute Force
        Time Complexity: O(n * k)
        Space Complexity: O(1)
        """
        
        # Approach 1: Brute Force
        def brute_force_solution(nums, k):
            # Implementation
            pass
        
        # Approach 2: Sliding Window
        def sliding_window_solution(nums, k):
            # Implementation
            pass
        
        # Return best solution
        return sliding_window_solution(nums, k)
```

## 📊 Complexity Analysis

| Pattern | Time Complexity | Space Complexity | When to Use |
|---------|----------------|------------------|-------------|
| Fixed Size Window | O(n) | O(1) | Fixed size subarrays |
| Variable Size Window | O(n) | O(1) | Variable size subarrays |
| Two Pointers | O(n) | O(1) | String/array problems |
| Prefix Sum | O(n) | O(n) | Subarray sums |
| Monotonic Queue | O(n) | O(k) | Max/min in window |
| Hash Map Window | O(n) | O(k) | Character counting |

## 🧪 Testing Strategy

Each solution includes:

1. **Multiple Test Cases**: Edge cases, normal cases, large inputs
2. **Benchmarking**: Compare different approaches
3. **Complexity Analysis**: Time and space complexity for each approach
4. **Error Handling**: Graceful handling of edge cases

## 📝 Common Problems in This Category

### Easy Problems
- [ ] Maximum Sum Subarray of Size K
- [ ] First Negative Number in Every Window of Size K
- [ ] Count Occurrences of Anagrams

### Medium Problems
- [ ] Longest Substring Without Repeating Characters (LeetCode #3)
- [ ] Minimum Size Subarray Sum (LeetCode #209)
- [ ] Longest Substring with At Most K Distinct Characters (LeetCode #340)
- [ ] Sliding Window Maximum (LeetCode #239)

### Hard Problems
- [ ] Minimum Window Substring (LeetCode #76)
- [ ] Substring with Concatenation of All Words (LeetCode #30)
- [ ] Longest Substring with At Most Two Distinct Characters (LeetCode #159)

## 🎯 Key Insights

1. **Sliding window is versatile**: Can solve many subarray and substring problems
2. **Fixed vs variable size**: Choose based on problem requirements
3. **Two pointers**: Often used to implement sliding window
4. **Monotonic structures**: Useful for maintaining max/min in window
5. **Hash maps**: Great for tracking frequencies and unique elements

## 🔧 How to Use This Template

1. **Copy the template**: Use `sliding_window_template.py` as a starting point
2. **Implement multiple approaches**: Start with brute force, then optimize
3. **Add test cases**: Include edge cases and large inputs
4. **Benchmark solutions**: Compare performance of different approaches
5. **Document complexity**: Always note time and space complexity

## 📚 Additional Resources

- [NeetCode Sliding Window Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53ldVwtstag2TL4HQhAnC8ATf)
- [LeetCode Sliding Window Tag](https://leetcode.com/tag/sliding-window/)

## 🎉 Tips for Success

1. **Start with brute force**: Always implement the obvious solution first
2. **Identify window type**: Fixed size vs variable size
3. **Use two pointers**: Often the key to efficient sliding window
4. **Consider data structures**: Hash maps, queues, stacks for specific needs
5. **Practice regularly**: Sliding window is a fundamental technique

---

**Happy Coding! 🚀**
