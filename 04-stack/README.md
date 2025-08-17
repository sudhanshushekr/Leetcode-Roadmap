# Stack - LeetCode Solutions

This folder contains solutions for stack problems following NeetCode's structured approach.

## 📁 File Structure

- `stack_template.py` - Template for creating new solutions
- `README.md` - This file with patterns and guidelines

## 🎯 Common Patterns

### 1. Monotonic Stack
- **Use when**: Need to find next greater/smaller element or maintain order
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(n) usually
- **Examples**: Next Greater Element, Daily Temperatures, Largest Rectangle in Histogram

### 2. Stack with Hash Map
- **Use when**: Need to track frequencies or mappings
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(n) usually
- **Examples**: Valid Parentheses, Remove Duplicate Letters

### 3. Two Stacks
- **Use when**: Need to maintain two different orders or properties
- **Time Complexity**: O(1) for operations typically
- **Space Complexity**: O(n) usually
- **Examples**: Min Stack, Queue using Stacks

### 4. Stack with Queue
- **Use when**: Need to combine stack and queue operations
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(n) usually
- **Examples**: Implement Queue using Stacks, Implement Stack using Queues

### 5. Parentheses Matching
- **Use when**: Need to validate or process parentheses expressions
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(n) usually
- **Examples**: Valid Parentheses, Generate Parentheses, Remove Invalid Parentheses

### 6. Expression Evaluation
- **Use when**: Need to evaluate mathematical expressions
- **Time Complexity**: O(n) typically
- **Space Complexity**: O(n) usually
- **Examples**: Evaluate Reverse Polish Notation, Basic Calculator

## 🚀 Solution Template Structure

Each solution follows this structure:

```python
class Solution:
    def problem_name(self, nums: List[int]) -> List[int]:
        """
        Problem: [Brief description]
        
        Example:
        Input: nums = [1, 2, 3, 4, 5]
        Output: [5, 4, 3, 2, 1]
        
        Approach 1: Brute Force
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        
        # Approach 1: Brute Force
        def brute_force_solution(nums):
            # Implementation
            pass
        
        # Approach 2: Stack
        def stack_solution(nums):
            # Implementation
            pass
        
        # Return best solution
        return stack_solution(nums)
```

## 📊 Complexity Analysis

| Pattern | Time Complexity | Space Complexity | When to Use |
|---------|----------------|------------------|-------------|
| Monotonic Stack | O(n) | O(n) | Next greater/smaller elements |
| Stack with Hash Map | O(n) | O(n) | Frequency tracking |
| Two Stacks | O(1) | O(n) | Dual properties |
| Stack with Queue | O(n) | O(n) | Combined operations |
| Parentheses Matching | O(n) | O(n) | Expression validation |
| Expression Evaluation | O(n) | O(n) | Mathematical expressions |

## 🧪 Testing Strategy

Each solution includes:

1. **Multiple Test Cases**: Edge cases, normal cases, large inputs
2. **Benchmarking**: Compare different approaches
3. **Complexity Analysis**: Time and space complexity for each approach
4. **Error Handling**: Graceful handling of edge cases

## 📝 Common Problems in This Category

### Easy Problems
- [ ] Valid Parentheses (LeetCode #20)
- [ ] Min Stack (LeetCode #155)
- [ ] Implement Queue using Stacks (LeetCode #232)
- [ ] Implement Stack using Queues (LeetCode #225)

### Medium Problems
- [ ] Next Greater Element (LeetCode #496)
- [ ] Daily Temperatures (LeetCode #739)
- [ ] Evaluate Reverse Polish Notation (LeetCode #150)
- [ ] Car Fleet (LeetCode #853)

### Hard Problems
- [ ] Largest Rectangle in Histogram (LeetCode #84)
- [ ] Basic Calculator (LeetCode #224)
- [ ] Remove Invalid Parentheses (LeetCode #301)
- [ ] Trapping Rain Water (LeetCode #42)

## 🎯 Key Insights

1. **Monotonic stacks are powerful**: Can solve many array problems efficiently
2. **LIFO property**: Last in, first out is key to stack operations
3. **Two stacks**: Often used to maintain different properties
4. **Parentheses**: Stack is natural for matching problems
5. **Expression evaluation**: Stack is perfect for postfix/infix conversion

## 🔧 How to Use This Template

1. **Copy the template**: Use `stack_template.py` as a starting point
2. **Implement multiple approaches**: Start with brute force, then optimize
3. **Add test cases**: Include edge cases and large inputs
4. **Benchmark solutions**: Compare performance of different approaches
5. **Document complexity**: Always note time and space complexity

## 📚 Additional Resources

- [NeetCode Stack Playlist](https://www.youtube.com/playlist?list=PLot-Xpze53ldVwtstag2TL4HQhAnC8ATf)
- [LeetCode Stack Tag](https://leetcode.com/tag/stack/)

## 🎉 Tips for Success

1. **Start with brute force**: Always implement the obvious solution first
2. **Think LIFO**: Remember last in, first out property
3. **Use monotonic stacks**: Great for next greater/smaller problems
4. **Consider two stacks**: When you need dual properties
5. **Practice regularly**: Stack is a fundamental data structure

---

**Happy Coding! 🚀**
