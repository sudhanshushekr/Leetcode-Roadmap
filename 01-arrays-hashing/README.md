# Arrays & Hashing - LeetCode Solutions

This folder contains solutions for arrays and hashing problems following NeetCode's structured approach.

## 📁 File Structure

- `arrays_hashing_template.py` - Template for creating new solutions
- `two_sum_solution.py` - Complete solution for Two Sum (LeetCode #1)
- `README.md` - This file with patterns and guidelines

## 🎯 Common Patterns

### 1. Hash Set/Map
- **Use when**: Need to track seen elements, find duplicates, or check existence
- **Time Complexity**: O(n) for lookup
- **Space Complexity**: O(n) for storage
- **Examples**: Two Sum, Contains Duplicate, Valid Anagram

### 2. Two Pointers
- **Use when**: Array is sorted, need to find pairs, or sliding window
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(1) usually
- **Examples**: Two Sum (sorted), Container With Most Water

### 3. Sliding Window
- **Use when**: Need to find subarrays with certain properties
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(1) or O(k) where k is window size
- **Examples**: Longest Substring Without Repeating Characters

### 4. Prefix Sum
- **Use when**: Need to calculate sum of subarrays efficiently
- **Time Complexity**: O(n) for preprocessing, O(1) for queries
- **Space Complexity**: O(n) for prefix array
- **Examples**: Subarray Sum Equals K

### 5. Sorting + Two Pointers
- **Use when**: Need to find pairs or triplets that sum to target
- **Time Complexity**: O(n log n) due to sorting
- **Space Complexity**: O(1) or O(n) depending on implementation
- **Examples**: 3Sum, 4Sum

## 🚀 Solution Template Structure

Each solution follows this structure:

```python
class Solution:
    def problem_name(self, nums: List[int]) -> int:
        """
        Problem: [Brief description]
        
        Example:
        Input: nums = [1, 2, 3, 4]
        Output: 10
        
        Approach 1: Brute Force
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        
        # Approach 1: Brute Force
        def brute_force_solution(nums):
            # Implementation
            pass
        
        # Approach 2: Optimized
        def optimized_solution(nums):
            # Implementation
            pass
        
        # Return best solution
        return optimized_solution(nums)
```

## 📊 Complexity Analysis

| Pattern | Time Complexity | Space Complexity | When to Use |
|---------|----------------|------------------|-------------|
| Hash Set/Map | O(n) | O(n) | Tracking seen elements |
| Two Pointers | O(n) | O(1) | Sorted arrays, pairs |
| Sliding Window | O(n) | O(1) | Subarray problems |
| Prefix Sum | O(n) | O(n) | Subarray sums |
| Sorting + Two Pointers | O(n log n) | O(1) | Finding pairs/triplets |

## 🧪 Testing Strategy

Each solution includes:

1. **Multiple Test Cases**: Edge cases, normal cases, large inputs
2. **Benchmarking**: Compare different approaches
3. **Complexity Analysis**: Time and space complexity for each approach
4. **Error Handling**: Graceful handling of edge cases

## 📝 Common Problems in This Category

### Easy Problems
- [ ] Two Sum (LeetCode #1)
- [ ] Contains Duplicate (LeetCode #217)
- [ ] Valid Anagram (LeetCode #242)
- [ ] Valid Parentheses (LeetCode #20)

### Medium Problems
- [ ] Group Anagrams (LeetCode #49)
- [ ] Top K Frequent Elements (LeetCode #347)
- [ ] Product of Array Except Self (LeetCode #238)
- [ ] Longest Consecutive Sequence (LeetCode #128)

### Hard Problems
- [ ] Longest Substring Without Repeating Characters (LeetCode #3)
- [ ] Substring with Concatenation of All Words (LeetCode #30)
- [ ] Minimum Window Substring (LeetCode #76)

## 🎯 Key Insights

1. **Hash maps are your friend**: Most array problems can be solved efficiently with hash maps
2. **Consider sorting**: If you need to find pairs, sorting can help
3. **Two pointers**: Very useful for sorted arrays and sliding windows
4. **Space vs Time trade-off**: Often you can trade space for time efficiency
5. **Edge cases**: Always consider empty arrays, single elements, duplicates

## 🔧 How to Use This Template

1. **Copy the template**: Use `arrays_hashing_template.py` as a starting point
2. **Implement multiple approaches**: Start with brute force, then optimize
3. **Add test cases**: Include edge cases and large inputs
4. **Benchmark solutions**: Compare performance of different approaches
5. **Document complexity**: Always note time and space complexity

## 📚 Additional Resources

- [NeetCode Arrays & Hashing Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53ldVwtstag2TL4HQhAnC8ATf)
- [LeetCode Arrays Tag](https://leetcode.com/tag/array/)
- [LeetCode Hash Table Tag](https://leetcode.com/tag/hash-table/)

## 🎉 Tips for Success

1. **Start with brute force**: Always implement the obvious solution first
2. **Look for patterns**: Many problems follow similar patterns
3. **Use hash maps**: They're often the key to optimization
4. **Consider edge cases**: Empty arrays, single elements, duplicates
5. **Practice regularly**: Arrays and hashing are fundamental concepts

---

**Happy Coding! 🚀**
