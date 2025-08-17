"""
Implement Trie (Prefix Tree) - LeetCode Problem 208
https://leetcode.com/problems/implement-trie-prefix-tree/

A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently 
store and retrieve keys in a dataset of strings. There are various applications of this 
data structure, such as autocomplete and spellchecker.

Implement the Trie class:
- Trie() Initializes the trie object.
- void insert(String word) Inserts the string word into the trie.
- boolean search(String word) Returns true if the string word is in the trie (i.e., was 
  inserted before), and false otherwise.
- boolean startsWith(String prefix) Returns true if there is a previously inserted string 
  word that has the prefix prefix, and false otherwise.

Example 1:
Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True

Constraints:
- 1 <= word.length, prefix.length <= 2000
- word and prefix consist only of lowercase English letters.
- At most 3 * 10^4 calls in total will be made to insert, search, and startsWith.
"""

from typing import Dict, Optional
import time


class TrieNode:
    """Node class for Trie data structure"""
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word = False


class Trie:
    """
    Trie (Prefix Tree) implementation with multiple approaches
    """
    
    def __init__(self):
        """
        Initialize the trie with a root node
        """
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie
        
        Algorithm:
        1. Start from root node
        2. For each character in word, traverse or create child node
        3. Mark the last node as end of word
        
        Time Complexity: O(m) - m is the length of the word
        Space Complexity: O(m) - Store all characters of the word
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """
        Search for a complete word in the trie
        
        Algorithm:
        1. Traverse the trie following the word characters
        2. Check if we reach the end and the node is marked as end of word
        
        Time Complexity: O(m) - m is the length of the word
        Space Complexity: O(1) - Only use pointers
        """
        node = self._get_node(word)
        return node is not None and node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        """
        Check if any word in the trie starts with the given prefix
        
        Algorithm:
        1. Traverse the trie following the prefix characters
        2. Return True if we can reach the end of prefix
        
        Time Complexity: O(m) - m is the length of the prefix
        Space Complexity: O(1) - Only use pointers
        """
        node = self._get_node(prefix)
        return node is not None
    
    def _get_node(self, word: str) -> Optional[TrieNode]:
        """
        Helper method to get the node at the end of a word/prefix
        
        Algorithm:
        1. Traverse the trie following the word characters
        2. Return the node at the end, or None if path doesn't exist
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node


class TrieArray:
    """
    Alternative Trie implementation using arrays instead of dictionaries
    """
    
    def __init__(self):
        """
        Initialize the trie with a root node using array representation
        """
        self.root = TrieArrayNode()
    
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie using array representation
        
        Algorithm:
        1. Use array of size 26 for each node (for lowercase letters)
        2. Map characters to array indices (a->0, b->1, etc.)
        3. Traverse or create nodes as needed
        
        Time Complexity: O(m) - m is the length of the word
        Space Complexity: O(m) - Store all characters of the word
        """
        node = self.root
        
        for char in word:
            index = ord(char) - ord('a')
            if not node.children[index]:
                node.children[index] = TrieArrayNode()
            node = node.children[index]
        
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """
        Search for a complete word in the trie
        
        Time Complexity: O(m) - m is the length of the word
        Space Complexity: O(1) - Only use pointers
        """
        node = self._get_node(word)
        return node is not None and node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        """
        Check if any word in the trie starts with the given prefix
        
        Time Complexity: O(m) - m is the length of the prefix
        Space Complexity: O(1) - Only use pointers
        """
        node = self._get_node(prefix)
        return node is not None
    
    def _get_node(self, word: str) -> Optional['TrieArrayNode']:
        """
        Helper method to get the node at the end of a word/prefix
        """
        node = self.root
        
        for char in word:
            index = ord(char) - ord('a')
            if not node.children[index]:
                return None
            node = node.children[index]
        
        return node


class TrieArrayNode:
    """Node class for array-based Trie implementation"""
    def __init__(self):
        self.children = [None] * 26  # Array for 26 lowercase letters
        self.is_end_of_word = False


class TrieHashSet:
    """
    Alternative Trie implementation using hash sets for storage
    """
    
    def __init__(self):
        """
        Initialize the trie using hash sets
        """
        self.words = set()
        self.prefixes = set()
    
    def insert(self, word: str) -> None:
        """
        Insert a word and all its prefixes
        
        Algorithm:
        1. Store the complete word
        2. Store all prefixes of the word
        
        Time Complexity: O(m²) - m is the length of the word (store all prefixes)
        Space Complexity: O(m²) - Store word and all its prefixes
        """
        self.words.add(word)
        
        # Add all prefixes
        for i in range(1, len(word) + 1):
            self.prefixes.add(word[:i])
    
    def search(self, word: str) -> bool:
        """
        Search for a complete word
        
        Time Complexity: O(1) - Hash set lookup
        Space Complexity: O(1) - No extra space
        """
        return word in self.words
    
    def startsWith(self, prefix: str) -> bool:
        """
        Check if any word starts with the prefix
        
        Time Complexity: O(1) - Hash set lookup
        Space Complexity: O(1) - No extra space
        """
        return prefix in self.prefixes


# Testing and Benchmarking

def test_trie_implementations():
    """Test all trie implementations with various test cases"""
    
    test_cases = [
        {
            "operations": [
                ("insert", "apple"),
                ("search", "apple", True),
                ("search", "app", False),
                ("startsWith", "app", True),
                ("insert", "app"),
                ("search", "app", True)
            ],
            "description": "Basic trie operations"
        },
        {
            "operations": [
                ("insert", "hello"),
                ("insert", "world"),
                ("search", "hello", True),
                ("search", "world", True),
                ("search", "hell", False),
                ("startsWith", "hell", True),
                ("startsWith", "wor", True),
                ("startsWith", "xyz", False)
            ],
            "description": "Multiple words"
        },
        {
            "operations": [
                ("insert", "a"),
                ("search", "a", True),
                ("startsWith", "a", True),
                ("search", "b", False),
                ("startsWith", "b", False)
            ],
            "description": "Single character"
        },
        {
            "operations": [
                ("search", "empty", False),
                ("startsWith", "empty", False)
            ],
            "description": "Empty trie"
        }
    ]
    
    implementations = [
        ("Dictionary Trie", Trie),
        ("Array Trie", TrieArray),
        ("Hash Set Trie", TrieHashSet)
    ]
    
    print("Trie Implementations - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        
        for impl_name, impl_class in implementations:
            try:
                trie = impl_class()
                all_passed = True
                
                for operation in test['operations']:
                    if operation[0] == "insert":
                        trie.insert(operation[1])
                    elif operation[0] == "search":
                        result = trie.search(operation[1])
                        expected = operation[2]
                        if result != expected:
                            all_passed = False
                    elif operation[0] == "startsWith":
                        result = trie.startsWith(operation[1])
                        expected = operation[2]
                        if result != expected:
                            all_passed = False
                
                status = "✅ PASS" if all_passed else "❌ FAIL"
                print(f"  {impl_name}: {status}")
                
            except Exception as e:
                print(f"  {impl_name}: ERROR - {e}")


def benchmark_trie_implementations():
    """Benchmark different trie implementations with larger input"""
    
    # Test data
    words = [
        "apple", "application", "apply", "appreciate", "approach",
        "book", "bookmark", "bookshelf", "bookstore", "booking",
        "computer", "compute", "computation", "computational", "computing",
        "data", "database", "dataset", "datatype", "datastructure"
    ]
    
    search_words = ["apple", "app", "book", "comput", "data", "xyz"]
    prefix_words = ["app", "book", "comput", "data", "xyz"]
    
    print(f"\nBenchmarking with {len(words)} words")
    print("=" * 50)
    
    implementations = [
        ("Dictionary Trie", Trie),
        ("Array Trie", TrieArray),
        ("Hash Set Trie", TrieHashSet)
    ]
    
    for impl_name, impl_class in implementations:
        try:
            trie = impl_class()
            
            # Time insertions
            start_time = time.time()
            for word in words:
                trie.insert(word)
            insert_time = time.time() - start_time
            
            # Time searches
            start_time = time.time()
            for word in search_words:
                trie.search(word)
            search_time = time.time() - start_time
            
            # Time prefix searches
            start_time = time.time()
            for prefix in prefix_words:
                trie.startsWith(prefix)
            prefix_time = time.time() - start_time
            
            print(f"{impl_name}:")
            print(f"  Insert: {insert_time:.6f} seconds")
            print(f"  Search: {search_time:.6f} seconds")
            print(f"  Prefix: {prefix_time:.6f} seconds")
            
        except Exception as e:
            print(f"{impl_name}: ERROR - {e}")


def complexity_analysis():
    """Print complexity analysis for all implementations"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Dictionary Trie", "O(m)", "O(m)", "O(m)", "O(m)", "Flexible, good performance"),
        ("Array Trie", "O(m)", "O(m)", "O(m)", "O(m)", "Fixed alphabet, efficient"),
        ("Hash Set Trie", "O(m²)", "O(m²)", "O(1)", "O(1)", "Simple, space inefficient")
    ]
    
    print(f"{'Implementation':<20} {'Insert':<10} {'Space':<10} {'Search':<10} {'Prefix':<10} {'Notes'}")
    print("-" * 80)
    
    for impl, insert, space, search, prefix, notes in analysis:
        print(f"{impl:<20} {insert:<10} {space:<10} {search:<10} {prefix:<10} {notes}")


if __name__ == "__main__":
    print("Trie (Prefix Tree) - Complete Implementation Analysis")
    print("=" * 60)
    
    # Run tests
    test_trie_implementations()
    
    # Run benchmarks
    benchmark_trie_implementations()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the Dictionary Trie for optimal performance.")
