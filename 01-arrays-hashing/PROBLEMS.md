# Arrays & Hashing Problems - Complete Solutions

This folder contains comprehensive solutions for classic arrays and hashing problems following NeetCode's structured approach.

## 📋 Problem List

### 1. Two Sum (LeetCode #1) - Easy
- **File**: `two_sum_solution.py`
- **Key Pattern**: Hash Map
- **Time Complexity**: O(n)
- **Space Complexity**: O(n)
- **Key Insight**: Use hash map to store seen elements and check for complement

### 2. Contains Duplicate (LeetCode #217) - Easy
- **File**: `contains_duplicate.py`
- **Key Pattern**: Hash Set
- **Time Complexity**: O(n)
- **Space Complexity**: O(n)
- **Key Insight**: Use set to track seen elements

### 3. Valid Anagram (LeetCode #242) - Easy
- **File**: `valid_anagram.py`
- **Key Pattern**: Character Counting
- **Time Complexity**: O(n)
- **Space Complexity**: O(1) - fixed character set
- **Key Insight**: Compare character frequencies using Counter

### 4. Group Anagrams (LeetCode #49) - Medium
- **File**: `group_anagrams.py`
- **Key Pattern**: Hash Map with Sorted Key
- **Time Complexity**: O(n * k * log k)
- **Space Complexity**: O(n * k)
- **Key Insight**: Use sorted string as key for grouping

### 5. Top K Frequent Elements (LeetCode #347) - Medium
- **File**: `top_k_frequent.py`
- **Key Pattern**: Heap (Priority Queue)
- **Time Complexity**: O(n + k log n)
- **Space Complexity**: O(n)
- **Key Insight**: Use min-heap to keep top k elements

### 6. Product of Array Except Self (LeetCode #238) - Medium
- **File**: `product_except_self.py`
- **Key Pattern**: Prefix and Suffix Arrays
- **Time Complexity**: O(n)
- **Space Complexity**: O(1) - excluding output
- **Key Insight**: Use prefix and suffix products without division

## 🎯 Common Patterns Covered

### Hash Set/Map Patterns
- **Two Sum**: Hash map for O(n) lookup
- **Contains Duplicate**: Hash set for tracking seen elements
- **Valid Anagram**: Character frequency counting
- **Group Anagrams**: Hash map with sorted keys

### Array Manipulation Patterns
- **Product Except Self**: Prefix and suffix arrays
- **Top K Elements**: Heap-based selection

### Advanced Techniques
- **Bucket Sort**: For frequency-based problems
- **Quickselect**: For selection problems
- **Logarithmic Approach**: For product problems

## 📊 Complexity Analysis Summary

| Problem | Best Time | Best Space | Key Data Structure |
|---------|-----------|------------|-------------------|
| Two Sum | O(n) | O(n) | Hash Map |
| Contains Duplicate | O(n) | O(n) | Hash Set |
| Valid Anagram | O(n) | O(1) | Counter/Array |
| Group Anagrams | O(n*k) | O(n*k) | Hash Map |
| Top K Frequent | O(n+k log n) | O(n) | Heap |
| Product Except Self | O(n) | O(1) | Prefix/Suffix |

## 🧪 Testing Strategy

Each solution includes:
- ✅ **Multiple Approaches**: From brute force to optimal
- ✅ **Comprehensive Testing**: Edge cases, normal cases, large inputs
- ✅ **Benchmarking**: Performance comparison between approaches
- ✅ **Complexity Analysis**: Detailed time and space complexity
- ✅ **Key Insights**: Important concepts and patterns

## 🚀 How to Use

1. **Study the Template**: Start with `arrays_hashing_template.py`
2. **Practice Problems**: Work through each problem file
3. **Compare Approaches**: See how different solutions perform
4. **Understand Patterns**: Focus on the key insights for each problem
5. **Test Your Understanding**: Run the test cases and benchmarks

## 📚 Learning Path

### Beginner Level
1. Two Sum - Learn hash map basics
2. Contains Duplicate - Understand hash sets
3. Valid Anagram - Master character counting

### Intermediate Level
4. Group Anagrams - Hash map with complex keys
5. Top K Frequent Elements - Heap data structure
6. Product of Array Except Self - Array manipulation

## 🎯 Key Takeaways

1. **Hash maps are fundamental**: Most array problems can be solved efficiently with hash maps
2. **Space vs Time trade-offs**: Often you can trade space for time efficiency
3. **Multiple approaches**: Always consider different solutions and their trade-offs
4. **Edge cases matter**: Empty arrays, single elements, duplicates, zeros
5. **Pattern recognition**: Many problems follow similar patterns

## 🔗 Related Problems

### Easy Problems to Try Next
- Valid Parentheses (LeetCode #20)
- Longest Common Prefix (LeetCode #14)
- Remove Duplicates from Sorted Array (LeetCode #26)

### Medium Problems to Try Next
- Longest Consecutive Sequence (LeetCode #128)
- Encode and Decode Strings (LeetCode #271)
- Design HashMap (LeetCode #706)

### Hard Problems to Try Next
- Longest Substring Without Repeating Characters (LeetCode #3)
- Minimum Window Substring (LeetCode #76)
- Substring with Concatenation of All Words (LeetCode #30)

---

**Happy Coding! 🚀**

*Each solution file contains detailed explanations, multiple approaches, comprehensive testing, and performance analysis following NeetCode's structured learning approach.*
