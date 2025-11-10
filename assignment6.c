1. 
flowchart 
    A([Start]) --> B[/Input: list, k, find_largest/]
    B --> C{Is list empty<br/>or k invalid?}
    C -->|Yes| D[Return None/Error]
    C -->|No| E[Initialize min_heap or max_heap]
    E --> F{find_largest?}
    F -->|Yes| G[Use min_heap of size k]
    F -->|No| H[Use max_heap of size k]
    G --> I[Iterate through list]
    H --> I
    I --> J{More elements?}
    J -->|Yes| K[Get current element]
    K --> L{Heap size < k?}
    L -->|Yes| M[Add element to heap]
    L -->|No| N{For largest:<br/>element > heap top?<br/>For smallest:<br/>element < heap top?}
    N -->|Yes| O[Remove heap top<br/>Add new element]
    N -->|No| P[Skip element]
    M --> J
    O --> J
    P --> J
    J -->|No| Q[Return heap top]
    Q --> R([End])
    D --> R
2.
flowchart TD
    subgraph MergeSort["MERGE SORT"]
        A1([Start]) --> B1{List length <= 1?}
        B1 -->|Yes| C1[Return list]
        B1 -->|No| D1[Find middle index]
        D1 --> E1[Split into left and right halves]
        E1 --> F1[Recursively sort left half]
        F1 --> G1[Recursively sort right half]
        G1 --> H1[Merge sorted halves]
        H1 --> I1[Initialize i=0, j=0, result]
        I1 --> J1{Both halves<br/>have elements?}
        J1 -->|Yes| K1{left[i] <= right[j]?}
        K1 -->|Yes| L1[Append left[i] to result<br/>i++]
        K1 -->|No| M1[Append right[j] to result<br/>j++]
        L1 --> J1
        M1 --> J1
        J1 -->|No| N1[Append remaining elements]
        N1 --> O1[Return merged list]
        O1 --> P1([End])
        C1 --> P1
    end
    
    subgraph QuickSort["QUICK SORT"]
        A2([Start]) --> B2{List length <= 1?}
        B2 -->|Yes| C2[Return list]
        B2 -->|No| D2[Choose pivot<br/>last element]
        D2 --> E2[Initialize left, right lists]
        E2 --> F2[Iterate through list<br/>except pivot]
        F2 --> G2{Element <= pivot?}
        G2 -->|Yes| H2[Add to left list]
        G2 -->|No| I2[Add to right list]
        H2 --> J2{More elements?}
        I2 --> J2
        J2 -->|Yes| F2
        J2 -->|No| K2[Recursively sort left]
        K2 --> L2[Recursively sort right]
        L2 --> M2[Concatenate:<br/>sorted_left + pivot + sorted_right]
        M2 --> N2[Return result]
        N2 --> O2([End])
        C2 --> O2
    end
3.
flowchart TD
    A([Start]) --> B[/Input: string s/]
    B --> C[Initialize cleaned_string = empty]
    C --> D[Iterate through each character]
    D --> E{Is character<br/>alphanumeric?}
    E -->|Yes| F[Add lowercase char<br/>to cleaned_string]
    E -->|No| G[Skip character]
    F --> H{More characters?}
    G --> H
    H -->|Yes| D
    H -->|No| I[Initialize left = 0<br/>right = length - 1]
    I --> J{left < right?}
    J -->|Yes| K{cleaned_string[left]<br/>== cleaned_string[right]?}
    K -->|Yes| L[left++<br/>right--]
    K -->|No| M[/Output: False/]
    L --> J
    J -->|No| N[/Output: True/]
    M --> O([End])
    N --> O
4.
def intersection(list1, list2):
    """Find intersection without using built-in set operations"""
    result = []
    for item in list1:
        if item in list2 and item not in result:
            result.append(item)
    return result

def union(list1, list2):
    """Find union without using built-in set operations"""
    result = list1.copy()
    for item in list2:
        if item not in result:
            result.append(item)
    return result

# Test
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

print(f"Intersection: {intersection(list1, list2)}")  # [4, 5]
print(f"Union: {union(list1, list2)}")  # [1, 2, 3, 4, 5, 6, 7, 8]

  5.
  flowchart TD
    subgraph Push["PUSH Operation"]
        A1([Start Push]) --> B1[Add element to q1]
        B1 --> C1([End Push])
    end
    
    subgraph Pop["POP Operation"]
        A2([Start Pop]) --> B2{Is q1 empty?}
        B2 -->|Yes| C2[Return None/Error]
        B2 -->|No| D2{q1 size == 1?}
        D2 -->|Yes| E2[Remove and return<br/>element from q1]
        D2 -->|No| F2[Move element from<br/>q1 to q2]
        F2 --> G2{q1 size > 1?}
        G2 -->|Yes| F2
        G2 -->|No| H2[Remove last element<br/>from q1 as result]
        H2 --> I2[Swap q1 and q2]
        I2 --> J2[Return result]
        J2 --> K2([End Pop])
        E2 --> K2
        C2 --> K2
    end
    
    subgraph Top["TOP Operation"]
        A3([Start Top]) --> B3{Is q1 empty?}
        B3 -->|Yes| C3[Return None/Error]
        B3 -->|No| D3[Move all but last<br/>element to q2]
        D3 --> E3[Peek last element in q1]
        E3 --> F3[Move last element to q2]
        F3 --> G3[Swap q1 and q2]
        G3 --> H3[Return peeked element]
        H3 --> I3([End Top])
        C3 --> I3
    end

6.
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        """Add node at the end"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def reverse(self):
        """Reverse the linked list"""
        prev = None
        current = self.head
        
        while current:
            next_node = current.next  # Store next
            current.next = prev        # Reverse the link
            prev = current             # Move prev forward
            current = next_node        # Move current forward
        
        self.head = prev
    
    def reverse_recursive(self, node=None):
        """Reverse linked list recursively"""
        if node is None:
            node = self.head
        
        # Base case
        if not node or not node.next:
            self.head = node
            return node
        
        # Recursive case
        new_head = self.reverse_recursive(node.next)
        node.next.next = node
        node.next = None
        return new_head
    
    def display(self):
        """Display the linked list"""
        current = self.head
        elements = []
        while current:
            elements.append(str(current.data))
            current = current.next
        print(" -> ".join(elements))

# Test
ll = LinkedList()
for i in range(1, 6):
    ll.append(i)

print("Original:")
ll.display()  # 1 -> 2 -> 3 -> 4 -> 5

ll.reverse()
print("Reversed:")
ll.display()  # 5 -> 4 -> 3 -> 2 -> 1

  
  7.
  flowchart TD
    A([Start]) --> B[/Input: arr, target, left, right/]
    B --> C{left > right?}
    C -->|Yes| D[Return -1<br/>Not Found]
    C -->|No| E[Calculate mid = left + right // 2]
    E --> F{arr[mid] == target?}
    F -->|Yes| G[Return mid]
    F -->|No| H{arr[mid] > target?}
    H -->|Yes| I[Recursively search<br/>left half<br/>left to mid-1]
    H -->|No| J[Recursively search<br/>right half<br/>mid+1 to right]
    I --> K[Return result]
    J --> K
    K --> L([End])
    D --> L
    G --> L


  8.
  flowchart TD
    A([Start]) --> B[/Input: array arr/]
    B --> C[max_sum = arr[0]<br/>current_sum = arr[0]]
    C --> D[Initialize i = 1]
    D --> E{i < length?}
    E -->|Yes| F[current_sum = max<br/>arr[i], current_sum + arr[i]]
    F --> G{current_sum > max_sum?}
    G -->|Yes| H[max_sum = current_sum]
    G -->|No| I[i++]
    H --> I
    I --> E
    E -->|No| J[/Output: max_sum/]
    J --> K([End])

  9.
  class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_balanced(root):
    """Check if binary tree is balanced"""
    def check_height(node):
        if not node:
            return 0
        
        left_height = check_height(node.left)
        if left_height == -1:
            return -1
        
        right_height = check_height(node.right)
        if right_height == -1:
            return -1
        
        if abs(left_height - right_height) > 1:
            return -1
        
        return max(left_height, right_height) + 1
    
    return check_height(root) != -1

# Test
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
print(is_balanced(root))  # True

  10.
  def rotate_array(arr, k):
    """Rotate array by k elements to the right"""
    n = len(arr)
    k = k % n  # Handle k > n
    
    # Method 1: Using slicing
    return arr[-k:] + arr[:-k]

def rotate_array_inplace(arr, k):
    """Rotate array in-place"""
    n = len(arr)
    k = k % n
    
    def reverse(start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    reverse(0, n - 1)      # Reverse entire array
    reverse(0, k - 1)      # Reverse first k elements
    reverse(k, n - 1)      # Reverse remaining elements
    
    return arr

# Test
arr = [1, 2, 3, 4, 5, 6, 7]
print(f"Rotated by 3: {rotate_array(arr.copy(), 3)}")  # [5, 6, 7, 1, 2, 3, 4]


  11.
  def permutations(s):
    """Find all permutations recursively"""
    if len(s) <= 1:
        return [s]
    
    result = []
    for i, char in enumerate(s):
        remaining = s[:i] + s[i+1:]
        for perm in permutations(remaining):
            result.append(char + perm)
    
    return result

# Test
print(permutations("abc"))  # ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

  12.
  from collections import Counter

def top_frequent_words(filename, top_n=5):
    """Count word frequency and display top N words"""
    try:
        with open(filename, 'r') as file:
            text = file.read().lower()
            # Remove punctuation and split
            words = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split()
            
            # Count frequencies
            word_count = Counter(words)
            
            # Get top N
            top_words = word_count.most_common(top_n)
            
            print(f"Top {top_n} most frequent words:")
            for word, count in top_words:
                print(f"{word}: {count}")
            
            return top_words
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []

# Test (create a sample file first)
# top_frequent_words("sample.txt", 5)


  13.
  def are_anagrams(str1, str2):
    """Check if two strings are anagrams"""
    # Remove spaces and convert to lowercase
    str1 = str1.replace(" ", "").lower()
    str2 = str2.replace(" ", "").lower()
    
    # Check if lengths are different
    if len(str1) != len(str2):
        return False
    
    # Count characters
    char_count = {}
    
    for char in str1:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in str2:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] < 0:
            return False
    
    return True

# Test
print(are_anagrams("listen", "silent"))  # True
print(are_anagrams("hello", "world"))    # False

  14.
  def compress_string(s):
    """Compress string like 'aaabbccccd' -> 'a3b2c4d1'"""
    if not s:
        return ""
    
    result = []
    count = 1
    current_char = s[0]
    
    for i in range(1, len(s)):
        if s[i] == current_char:
            count += 1
        else:
            result.append(current_char + str(count))
            current_char = s[i]
            count = 1
    
    # Add the last character
    result.append(current_char + str(count))
    
    return ''.join(result)

# Test
print(compress_string("aaabbccccd"))  # a3b2c4d1
print(compress_string("aabbcc"))      # a2b2c2

  15.
  def reverse_words(sentence):
    """Reverse order of words preserving punctuation"""
    import re
    
    # Extract words and non-words
    tokens = re.findall(r'\w+|\W+', sentence)
    
    # Separate words and non-words
    words = [t for t in tokens if t.strip() and t.isalnum()]
    
    # Reverse words
    words.reverse()
    
    # Rebuild sentence
    word_index = 0
    result = []
    for token in tokens:
        if token.strip() and token.isalnum():
            result.append(words[word_index])
            word_index += 1
        else:
            result.append(token)
    
    return ''.join(result)

# Test
print(reverse_words("Hello, world! How are you?"))
# Output: "you are How world! Hello,"


  16.
  def fibonacci_memo(n, memo=None):
    """Find nth Fibonacci number using memoization"""
    if memo is None:
        memo = {}
    
    # Base cases
    if n <= 1:
        return n
    
    # Check if already computed
    if n in memo:
        return memo[n]
    
    # Compute and store
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# Test
print(f"10th Fibonacci: {fibonacci_memo(10)}")  # 55
print(f"20th Fibonacci: {fibonacci_memo(20)}")  # 6765

  17.
  def all_subsets(lst):
    """Generate all subsets of a list"""
    if not lst:
        return [[]]
    
    first = lst[0]
    rest_subsets = all_subsets(lst[1:])
    
    # Add subsets with first element
    new_subsets = [[first] + subset for subset in rest_subsets]
    
    return rest_subsets + new_subsets

# Alternative iterative approach
def all_subsets_iterative(lst):
    """Generate all subsets iteratively"""
    result = [[]]
    
    for num in lst:
        result += [subset + [num] for subset in result]
    
    return result

# Test
print(all_subsets([1, 2, 3]))
# [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]


18.
def evaluate_postfix(expression):
    """Evaluate a postfix expression"""
    stack = []
    operators = {'+', '-', '*', '/', '^'}
    
    for token in expression.split():
        if token not in operators:
            stack.append(float(token))
        else:
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                result = a / b
            elif token == '^':
                result = a ** b
            
            stack.append(result)
    
    return stack[0]

# Test
print(evaluate_postfix("3 4 + 2 * 7 /"))  # ((3 + 4) * 2) / 7 = 2.0
print(evaluate_postfix("5 1 2 + 4 * + 3 -"))  # 5 + ((1 + 2) * 4) - 3 = 14.0


  19.
def find_pairs_with_sum(arr, target_sum):
    """Find all pairs that sum to target"""
    pairs = []
    seen = set()
    
    for num in arr:
        complement = target_sum - num
        if complement in seen:
            pairs.append((min(num, complement), max(num, complement)))
        seen.add(num)
    
    # Remove duplicates
    return list(set(pairs))

def find_all_pairs(arr):
    """Find all possible pairs"""
    pairs = []
    n = len(arr)
    
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((arr[i], arr[j]))
    
    return pairs

# Test
arr = [1, 5, 7, -1, 5]
print(f"Pairs with sum 6: {find_pairs_with_sum(arr, 6)}")  # [(1, 5), (7, -1)]
print(f"All pairs: {find_all_pairs([1, 2, 3])}")  # [(1, 2), (1, 3), (2, 3)]



20.
  graph TD
    Start["Input: [1, [2, 3], [4, [5, 6]], 7]"]
    
    Start --> L1["Level 1: Process element 1"]
    L1 --> L1R["Result: [1]"]
    
    L1R --> L2["Level 1: Process [2, 3]"]
    L2 --> L2A["Is list? YES - Recurse"]
    L2A --> L2B["Flatten [2, 3]"]
    L2B --> L2C["Process 2: add to result"]
    L2C --> L2D["Process 3: add to result"]
    L2D --> L2R["Result: [1, 2, 3]"]
    
    L2R --> L3["Level 1: Process [4, [5, 6]]"]
    L3 --> L3A["Is list? YES - Recurse"]
    L3A --> L3B["Flatten [4, [5, 6]]"]
    L3B --> L3C["Process 4: add to result"]
    L3C --> L3D["Process [5, 6]"]
    L3D --> L3E["Is list? YES - Recurse"]
    L3E --> L3F["Flatten [5, 6]"]
    L3F --> L3G["Process 5: add to result"]
    L3G --> L3H["Process 6: add to result"]
    L3H --> L3R["Result: [1, 2, 3, 4, 5, 6]"]
    
    L3R --> L4["Level 1: Process element 7"]
    L4 --> L4R["Result: [1, 2, 3, 4, 5, 6, 7]"]
    
    L4R --> Final["Final Output: [1, 2, 3, 4, 5, 6, 7]"]
    
    style Start fill:#FFE4B5
    style Final fill:#90EE90
    style L1R fill:#E0F0FF
    style L2R fill:#E0F0FF
    style L3R fill:#E0F0FF
    style L4R fill:#E0F0FF
    style L2A fill:#FFB6C1
    style L3A fill:#FFB6C1
    style L3E fill:#FFB6C1
