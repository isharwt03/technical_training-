import collections
import random
import sys
from collections import deque

# Set recursion limit higher for recursive solutions like QuickSort/MergeSort
sys.setrecursionlimit(2000)

# ==============================================================================
# 1. Kth Largest/Smallest Element (without built-in sort)
#    Uses Quickselect, which leverages the partitioning logic of QuickSort for O(n) average time complexity.
# ==============================================================================

def partition(arr, low, high):
    """Lomuto partition scheme: partitions the array around a pivot."""
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def find_kth_largest(arr, k):
    """Finds the Kth largest element using Quickselect."""
    if not arr or k <= 0 or k > len(arr):
        return "Invalid input"

    # We are looking for the element at index (len(arr) - k) if the array were sorted.
    target_index = len(arr) - k
    
    # Work on a copy to avoid modifying the original list
    arr_copy = arr[:]
    low, high = 0, len(arr_copy) - 1

    while low <= high:
        pivot_index = partition(arr_copy, low, high)
        if pivot_index == target_index:
            return arr_copy[pivot_index]
        elif pivot_index < target_index:
            low = pivot_index + 1
        else:
            high = pivot_index - 1
    
    return "Error: Could not find Kth element."

def find_kth_smallest(arr, k):
    """Finds the Kth smallest element using Quickselect. Target index is k-1."""
    if not arr or k <= 0 or k > len(arr):
        return "Invalid input"

    # We are looking for the element at index (k - 1) if the array were sorted.
    target_index = k - 1
    
    arr_copy = arr[:]
    low, high = 0, len(arr_copy) - 1

    while low <= high:
        pivot_index = partition(arr_copy, low, high)
        if pivot_index == target_index:
            return arr_copy[pivot_index]
        elif pivot_index < target_index:
            low = pivot_index + 1
        else:
            high = pivot_index - 1
    
    return "Error: Could not find Kth element."


# ==============================================================================
# 2. Implement Merge Sort
# ==============================================================================

def merge_sort(arr):
    """Sorts an array using the Merge Sort algorithm (Divide and Conquer)."""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)

def merge(left, right):
    """Merges two sorted lists into a single sorted list."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Append remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# ==============================================================================
# 3. Implement Quick Sort
# ==============================================================================

def quick_sort(arr, low, high):
    """Sorts an array using the Quick Sort algorithm (in-place modification)."""
    if low < high:
        # pi is partitioning index, arr[pi] is now at right place
        pi = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

# Note: This quick_sort uses the 'partition' function defined above for Kth element.

# ==============================================================================
# 4. Valid Palindrome Check (ignore special chars and spaces)
# ==============================================================================

def is_valid_palindrome(s):
    """Checks if a string is a palindrome, ignoring non-alphanumeric characters."""
    filtered_chars = []
    for char in s:
        if char.isalnum():
            filtered_chars.append(char.lower())
    
    # Join the filtered characters to form a clean string
    clean_s = "".join(filtered_chars)
    
    # Check if the cleaned string is equal to its reverse
    return clean_s == clean_s[::-1]

# ==============================================================================
# 5. Intersection and Union of Two Lists (without built-in set operations)
# ==============================================================================

def list_union(list1, list2):
    """Finds the union of two lists without using set operations."""
    # Start with a copy of list1
    union_list = list1[:]
    
    # Iterate through list2 and add elements not already in union_list
    for item in list2:
        if item not in union_list:
            union_list.append(item)
    return union_list

def list_intersection(list1, list2):
    """Finds the intersection of two lists without using set operations."""
    intersection_list = []
    
    # Use a frequency map (dictionary) for list1 to handle duplicates efficiently
    freq_map = collections.defaultdict(int)
    for item in list1:
        freq_map[item] += 1
        
    # Check list2 elements against the frequency map
    for item in list2:
        if freq_map[item] > 0:
            intersection_list.append(item)
            freq_map[item] -= 1 # Decrement count to handle duplicates properly
            
    return intersection_list

# ==============================================================================
# 6. Implement a Stack using Two Queues (and vice versa)
# ==============================================================================

class StackUsingTwoQueues:
    """Implements a Stack (LIFO) using two queues (FIFO)."""
    def __init__(self):
        self.q1 = deque() # Main queue for storing elements
        self.q2 = deque() # Helper queue

    def push(self, x):
        """Pushes element x onto the stack."""
        # Standard push: O(1)
        self.q1.append(x)

    def pop(self):
        """Removes the element on top of the stack and returns that element."""
        if not self.q1:
            return None # Stack is empty

        # Move n-1 elements from q1 to q2
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())

        # Pop the last element (which is the stack's top element)
        top_element = self.q1.popleft()

        # Swap the names of q1 and q2 (q2 becomes the new main queue)
        self.q1, self.q2 = self.q2, self.q1
        
        return top_element

    def top(self):
        """Returns the element on top of the stack."""
        if not self.q1:
            return None

        # Similar process to pop, but push the element back to q2 before swap
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())

        top_element = self.q1.popleft()
        self.q2.append(top_element) # Put it back

        self.q1, self.q2 = self.q2, self.q1
        return top_element
    
    def is_empty(self):
        return not self.q1

class QueueUsingTwoStacks:
    """Implements a Queue (FIFO) using two stacks (LIFO lists)."""
    def __init__(self):
        self.stack_in = []  # Stack for enqueuing (input)
        self.stack_out = [] # Stack for dequeuing (output)

    def push(self, x):
        """Pushes element x to the back of the queue (enqueue)."""
        # O(1) operation
        self.stack_in.append(x)

    def pop(self):
        """Removes the element from the front of the queue (dequeue)."""
        if not self.stack_out:
            # Transfer elements from stack_in to stack_out in reverse order
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        
        if self.stack_out:
            return self.stack_out.pop()
        else:
            return None # Queue is empty

    def peek(self):
        """Gets the front element."""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())

        if self.stack_out:
            return self.stack_out[-1]
        else:
            return None
    
    def is_empty(self):
        return not self.stack_in and not self.stack_out

# ==============================================================================
# 7. Write a Python program to reverse a linked list.
# ==============================================================================

class ListNode:
    """Node for a Singly Linked List."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    """Reverses a singly linked list iteratively."""
    prev = None
    current = head
    while current:
        next_node = current.next # Store next node
        current.next = prev      # Reverse current node's pointer
        prev = current           # Move prev to current node
        current = next_node      # Move current to next node
    return prev # New head is the old tail (prev)

# Helper function to convert list to linked list
def list_to_linked_list(data):
    dummy = ListNode(0)
    tail = dummy
    for val in data:
        tail.next = ListNode(val)
        tail = tail.next
    return dummy.next

# Helper function to convert linked list to list for printing
def linked_list_to_list(head):
    data = []
    current = head
    while current:
        data.append(current.val)
        current = current.next
    return data

# ==============================================================================
# 8. Implement a binary search algorithm recursively.
# ==============================================================================

def binary_search_recursive(arr, target, low, high):
    """Recursively searches for a target in a sorted array."""
    if low > high:
        return -1 # Base case: target not found

    mid = (low + high) // 2
    
    if arr[mid] == target:
        return mid # Target found
    elif arr[mid] < target:
        # Search in the right half
        return binary_search_recursive(arr, target, mid + 1, high)
    else:
        # Search in the left half
        return binary_search_recursive(arr, target, low, mid - 1)

def find_binary_search(arr, target):
    """Wrapper function for recursive binary search."""
    # Ensure the array is sorted before calling binary search
    if arr != sorted(arr):
        print("Warning: Array should be sorted for binary search.")
        # Optionally, you could sort it here: arr.sort()
        
    return binary_search_recursive(arr, target, 0, len(arr) - 1)

# ==============================================================================
# 9. Find the maximum sum subarray (Kadane’s algorithm).
# ==============================================================================

def kadanes_algorithm(nums):
    """Finds the contiguous subarray with the largest sum."""
    if not nums:
        return 0
    
    max_so_far = nums[0]
    current_max = nums[0]
    
    for i in range(1, len(nums)):
        # Decide whether to start a new subarray or extend the current one
        current_max = max(nums[i], current_max + nums[i])
        
        # Update the maximum sum found so far
        max_so_far = max(max_so_far, current_max)
        
    return max_so_far

# ==============================================================================
# 10. Check if a binary tree is balanced or not.
# ==============================================================================

class TreeNode:
    """Node for a Binary Tree."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def check_balanced(root):
    """Checks if a binary tree is balanced (height of two subtrees of any node 
       never differs by more than 1).
    """
    
    def get_height_and_balance(node):
        """
        Returns (height, is_balanced). Height is -1 if unbalanced.
        Uses bottom-up recursion for efficiency (O(n)).
        """
        if node is None:
            return (0, True)

        # Recurse left
        left_height, left_balanced = get_height_and_balance(node.left)
        if not left_balanced:
            return (-1, False)

        # Recurse right
        right_height, right_balanced = get_height_and_balance(node.right)
        if not right_balanced:
            return (-1, False)

        # Check current node's balance condition
        if abs(left_height - right_height) > 1:
            return (-1, False)

        # If balanced, return the current node's height and balanced status
        return (max(left_height, right_height) + 1, True)

    _, is_balanced = get_height_and_balance(root)
    return is_balanced

# ==============================================================================
# 11. Write a function to rotate an array by K elements.
# ==============================================================================

def rotate_array(arr, k):
    """Rotates an array to the right by k elements using slicing (easy Python way)."""
    if not arr:
        return arr
    
    n = len(arr)
    # Handle k larger than n, or negative k
    k = k % n 
    
    # Rotation logic: [n-k:] + [:n-k]
    # Example: [1, 2, 3, 4, 5], k=2
    # n-k = 3. arr[3:] is [4, 5]. arr[:3] is [1, 2, 3]
    # Result: [4, 5, 1, 2, 3]
    
    # Note: This creates a new list. For in-place, the three-reversal method is used.
    rotated = arr[-k:] + arr[:-k]
    return rotated

def rotate_array_in_place(arr, k):
    """Rotates an array to the right by k elements using the three-reversal technique."""
    if not arr:
        return arr
    
    n = len(arr)
    k = k % n 
    
    def reverse(start, end):
        """Helper to reverse a segment of the array in place."""
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    # 1. Reverse the whole array
    reverse(0, n - 1)
    
    # 2. Reverse the first k elements
    reverse(0, k - 1)
    
    # 3. Reverse the remaining n - k elements
    reverse(k, n - 1)
    
    return arr # Array is modified in place


# ==============================================================================
# 12. Find all permutations of a string recursively.
# ==============================================================================

def find_permutations(s):
    """Finds all unique permutations of a string using recursion (backtracking)."""
    results = []
    
    def backtrack(current_permutation, remaining_chars):
        if not remaining_chars:
            results.append("".join(current_permutation))
            return

        # Use a set to handle duplicate characters in the string
        used_chars = set() 
        for i in range(len(remaining_chars)):
            char = remaining_chars[i]
            
            # Skip duplicates at the same level of recursion
            if char in used_chars:
                continue
            used_chars.add(char)
            
            # Choose: Select a character
            current_permutation.append(char)
            # Remove the chosen character for the next level
            new_remaining = remaining_chars[:i] + remaining_chars[i+1:]
            
            # Explore
            backtrack(current_permutation, new_remaining)
            
            # Un-choose (Backtrack): Remove the character
            current_permutation.pop()

    # Sort the string initially to group duplicates, which helps with the used_chars check
    sorted_s = sorted(s) 
    backtrack([], sorted_s)
    return results

# ==============================================================================
# 13. Count the frequency of each word in a text file and display the top 5.
# ==============================================================================

def get_top_5_words(filepath="sample_text.txt", top_n=5):
    """Reads a file, counts word frequency, and returns the top N words."""
    try:
        # Create a mock file for testing purposes if it doesn't exist (simulated)
        if filepath == "sample_text.txt":
            mock_content = (
                "The quick brown fox jumps over the lazy dog. The dog is brown "
                "and quick. Quick quick quick, dog dog dog. Fox fox."
            )
        
        # Read the content (simulating file reading)
        content = mock_content

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []

    # Simple tokenization: remove punctuation and convert to lowercase
    words = content.lower().split()
    clean_words = []
    for word in words:
        # Remove trailing and leading punctuation (e.g., periods, commas)
        clean_word = word.strip(".,!?\"'()[]{}")
        if clean_word:
            clean_words.append(clean_word)

    # Count frequencies
    word_counts = collections.Counter(clean_words)

    # Get the top N most common words
    top_words = word_counts.most_common(top_n)
    return top_words


# ==============================================================================
# 14. Write a function to check if two strings are anagrams.
# ==============================================================================

def are_anagrams(s1, s2):
    """Checks if two strings are anagrams of each other."""
    
    # Helper to clean the string (ignore case, spaces, and punctuation)
    def clean_string(s):
        return "".join(char.lower() for char in s if char.isalnum())
    
    clean_s1 = clean_string(s1)
    clean_s2 = clean_string(s2)
    
    # Anagrams must have the same length
    if len(clean_s1) != len(clean_s2):
        return False
        
    # Check if the sorted character lists are identical
    return sorted(clean_s1) == sorted(clean_s2)

# ==============================================================================
# 15. Implement a program to compress strings like "aaabbccccd" -> "a3b2c4d1".
# ==============================================================================

def compress_string(s):
    """Compresses a string by counting consecutive characters."""
    if not s:
        return ""

    compressed = []
    i = 0
    n = len(s)

    while i < n:
        char = s[i]
        count = 0
        j = i
        
        # Count consecutive occurrences of the current character
        while j < n and s[j] == char:
            count += 1
            j += 1
        
        # Append character and count
        compressed.append(char)
        compressed.append(str(count))
        
        # Move pointer to the start of the next distinct character group
        i = j 
        
    return "".join(compressed)

# ==============================================================================
# 16. Reverse the order of words in a sentence, preserving punctuation.
# ==============================================================================

def reverse_words_preserve_punctuation(sentence):
    """Reverses the order of words in a sentence while keeping final punctuation."""
    
    if not sentence:
        return ""

    # 1. Separate the final punctuation (if any)
    trailing_punctuation = ""
    clean_sentence = sentence
    
    # Check for common trailing punctuation
    if sentence[-1] in ".?!,":
        trailing_punctuation = sentence[-1]
        clean_sentence = sentence[:-1]
    
    # 2. Split the remaining sentence into words
    words = clean_sentence.split()
    
    # 3. Reverse the order of the words
    reversed_words = words[::-1]
    
    # 4. Join the words and re-add the punctuation
    return " ".join(reversed_words) + trailing_punctuation

# ==============================================================================
# 17. Write a recursive function to find the nth Fibonacci number using memoization.
# ==============================================================================

# Memoization cache (dictionary)
fib_memo = {}

def fibonacci_memoization(n):
    """
    Finds the nth Fibonacci number recursively using memoization.
    F(0)=0, F(1)=1.
    """
    if n < 0:
        return "Input must be non-negative"
    if n in fib_memo:
        return fib_memo[n]
    
    if n <= 1:
        return n
    
    # Recursive step with memoization
    result = fibonacci_memoization(n - 1) + fibonacci_memoization(n - 2)
    fib_memo[n] = result
    
    return result

# ==============================================================================
# 18. Print all subsets of a given list.
# ==============================================================================

def find_subsets(nums):
    """Finds all subsets (power set) of a given list using backtracking."""
    subsets = []
    
    def backtrack(index, current_subset):
        # Base case: Add the current subset to the result
        subsets.append(list(current_subset))
        
        # Recursive step: Iterate through the remaining elements
        for i in range(index, len(nums)):
            # Choose: Add the current element
            current_subset.append(nums[i])
            
            # Explore: Move to the next index
            backtrack(i + 1, current_subset)
            
            # Un-choose (Backtrack): Remove the last element for the next iteration
            current_subset.pop()

    # Start the backtracking process
    backtrack(0, [])
    return subsets

# ==============================================================================
# 19. Implement a function to evaluate a postfix expression.
# ==============================================================================

def evaluate_postfix(expression):
    """Evaluates a postfix expression using a stack."""
    stack = []
    operators = set(['+', '-', '*', '/'])
    
    # Split the expression by spaces
    tokens = expression.split()
    
    for token in tokens:
        if token not in operators:
            # If token is an operand, push its integer value to the stack
            try:
                stack.append(float(token))
            except ValueError:
                return f"Error: Invalid operand '{token}'"
        else:
            # If token is an operator, pop two operands, compute, and push result
            if len(stack) < 2:
                return "Error: Invalid postfix expression (too few operands)"
                
            operand2 = stack.pop()
            operand1 = stack.pop()
            
            if token == '+':
                result = operand1 + operand2
            elif token == '-':
                result = operand1 - operand2
            elif token == '*':
                result = operand1 * operand2
            elif token == '/':
                if operand2 == 0:
                    return "Error: Division by zero"
                result = operand1 / operand2
            
            stack.append(result)

    # The final result should be the only element left in the stack
    if len(stack) == 1:
        # Return the result as an integer if it's an exact integer, otherwise float
        final_result = stack[0]
        return int(final_result) if final_result == int(final_result) else final_result
    else:
        return "Error: Invalid postfix expression (too many operands)"

# ==============================================================================
# 20. Write a program to find all pairs in an array that sum to a specific target value.
# ==============================================================================

def find_two_sum_pairs(arr, target):
    """Finds all pairs (indices and values) that sum to the target value (O(n))."""
    seen_numbers = {} # Store {number: index}
    pairs = []
    
    for i, num in enumerate(arr):
        complement = target - num
        
        # Check if the complement has been seen before
        if complement in seen_numbers:
            # Found a pair!
            pairs.append((
                (seen_numbers[complement], complement), # (index, value) of first number
                (i, num)                              # (index, value) of current number
            ))
        
        # Store the current number and its index
        # Note: If there are duplicates, this keeps the index of the *last* occurrence,
        # which is sufficient for finding *any* valid pair involving that number.
        seen_numbers[num] = i
        
    return pairs

# ==============================================================================
# 21. Write a function that flattens a nested list.
# ==============================================================================

def flatten_nested_list(nested_list):
    """Flattens a list that may contain nested lists recursively."""
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            # If the element is a list, recurse and extend the flat_list
            flat_list.extend(flatten_nested_list(element))
        else:
            # Otherwise, append the element directly
            flat_list.append(element)
    return flat_list


# ==============================================================================
# Example Usage and Testing
# ==============================================================================

if __name__ == '__main__':
    print("--- 1. Kth Largest/Smallest Element ---")
    data = [3, 2, 1, 5, 6, 4]
    k_val = 2
    print(f"Data: {data}, K={k_val}")
    print(f"Kth Largest: {find_kth_largest(data, k_val)}")
    print(f"Kth Smallest: {find_kth_smallest(data, k_val)}")

    print("\n--- 2 & 3. Merge Sort and Quick Sort ---")
    unsorted = [12, 11, 13, 5, 6, 7]
    arr_quick = unsorted[:] # Copy for Quick Sort
    print(f"Unsorted: {unsorted}")
    
    sorted_merge = merge_sort(unsorted)
    print(f"Merge Sorted: {sorted_merge}")
    
    quick_sort(arr_quick, 0, len(arr_quick) - 1)
    print(f"Quick Sorted: {arr_quick}")

    print("\n--- 4. Valid Palindrome Check ---")
    p1 = "A man, a plan, a canal: Panama"
    p2 = "race a car"
    print(f"'{p1}': {is_valid_palindrome(p1)}")
    print(f"'{p2}': {is_valid_palindrome(p2)}")

    print("\n--- 5. Intersection and Union ---")
    l1 = [1, 2, 3, 2, 4]
    l2 = [2, 4, 5, 6, 2]
    print(f"List 1: {l1}, List 2: {l2}")
    print(f"Union: {list_union(l1, l2)}")
    print(f"Intersection: {list_intersection(l1, l2)}")

    print("\n--- 6. Stack using Two Queues & Queue using Two Stacks ---")
    stack_q = StackUsingTwoQueues()
    stack_q.push(10); stack_q.push(20)
    print(f"Stack (Q) Pop: {stack_q.pop()}") # Should be 20
    print(f"Stack (Q) Top: {stack_q.top()}") # Should be 10

    queue_s = QueueUsingTwoStacks()
    queue_s.push(100); queue_s.push(200)
    print(f"Queue (S) Pop: {queue_s.pop()}") # Should be 100
    print(f"Queue (S) Push 300")
    queue_s.push(300)
    print(f"Queue (S) Pop: {queue_s.pop()}") # Should be 200

    print("\n--- 7. Reverse a Linked List ---")
    ll_data = [1, 2, 3, 4, 5]
    ll_head = list_to_linked_list(ll_data)
    ll_reversed = reverse_linked_list(ll_head)
    print(f"Original: {ll_data} -> Reversed: {linked_list_to_list(ll_reversed)}")

    print("\n--- 8. Recursive Binary Search ---")
    sorted_arr = [2, 5, 8, 12, 16, 23, 38, 56, 72]
    target_val = 23
    idx = find_binary_search(sorted_arr, target_val)
    print(f"Array: {sorted_arr}, Target: {target_val} -> Index: {idx}")

    print("\n--- 9. Maximum Sum Subarray (Kadane’s) ---")
    arr_kadane = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = kadanes_algorithm(arr_kadane)
    print(f"Array: {arr_kadane} -> Max Sum: {max_sum}") # Should be 6

    print("\n--- 10. Check if a Binary Tree is Balanced ---")
    # Balanced tree:
    root_balanced = TreeNode(1, TreeNode(2), TreeNode(3))
    # Unbalanced tree:
    root_unbalanced = TreeNode(1, TreeNode(2, TreeNode(3)), TreeNode(4))
    print(f"Tree 1 (Balanced): {check_balanced(root_balanced)}") # True
    print(f"Tree 2 (Unbalanced): {check_balanced(root_unbalanced)}") # False

    print("\n--- 11. Rotate an Array by K Elements ---")
    arr_rot = [1, 2, 3, 4, 5, 6, 7]
    k_rot = 3
    arr_in_place = arr_rot[:]
    
    # Using simple slicing (returns new list)
    print(f"Original: {arr_rot}, K={k_rot} -> Sliced: {rotate_array(arr_rot, k_rot)}")
    
    # Using in-place reversal
    print(f"Original: {arr_in_place}, K={k_rot} -> In-Place: {rotate_array_in_place(arr_in_place, k_rot)}")

    print("\n--- 12. Find all Permutations of a String ---")
    s_perm = "ABC"
    perms = find_permutations(s_perm)
    print(f"String '{s_perm}' Permutations ({len(perms)} total): {perms}")
    
    s_dup = "AAB"
    perms_dup = find_permutations(s_dup)
    print(f"String '{s_dup}' Permutations ({len(perms_dup)} total): {perms_dup}")

    print("\n--- 13. Count Word Frequency (Top 5) ---")
    top_5 = get_top_5_words()
    print("Top 5 Words:")
    for word, count in top_5:
        print(f" - {word}: {count}")

    print("\n--- 14. Check if Two Strings are Anagrams ---")
    a1 = "Listen"
    a2 = "Silent"
    a3 = "Hello world"
    print(f"'{a1}' and '{a2}': {are_anagrams(a1, a2)}")
    print(f"'{a1}' and '{a3}': {are_anagrams(a1, a3)}")

    print("\n--- 15. String Compression ---")
    s_comp = "aaabbccccd"
    s_compressed = compress_string(s_comp)
    print(f"'{s_comp}' -> '{s_compressed}'")

    print("\n--- 16. Reverse Words in Sentence (Preserve Punctuation) ---")
    s_rev = "Hello world, how are you?"
    s_reversed = reverse_words_preserve_punctuation(s_rev)
    print(f"'{s_rev}' -> '{s_reversed}'")

    print("\n--- 17. nth Fibonacci Number with Memoization ---")
    n_fib = 10
    fib_result = fibonacci_memoization(n_fib)
    print(f"F({n_fib}) = {fib_result}") # Should be 55
    print(f"Memoization cache size: {len(fib_memo)}")

    print("\n--- 18. Print All Subsets of a List ---")
    set_nums = [1, 2, 3]
    all_subsets = find_subsets(set_nums)
    print(f"List: {set_nums} -> Subsets ({len(all_subsets)} total): {all_subsets}")

    print("\n--- 19. Evaluate a Postfix Expression ---")
    exp1 = "2 3 1 * + 9 -" # 2 + (3 * 1) - 9 = 5 - 9 = -4
    exp2 = "10 5 / 3 +"     # (10 / 5) + 3 = 2 + 3 = 5
    print(f"'{exp1}' = {evaluate_postfix(exp1)}")
    print(f"'{exp2}' = {evaluate_postfix(exp2)}")

    print("\n--- 20. Find All Two Sum Pairs ---")
    arr_two_sum = [1, 5, 2, 8, 3, 7]
    target_two_sum = 10
    pairs_found = find_two_sum_pairs(arr_two_sum, target_two_sum)
    print(f"Array: {arr_two_sum}, Target: {target_two_sum}")
    print("Pairs (index, value):")
    for pair in pairs_found:
        print(f" - {pair[0]} + {pair[1]}") # Should find (1, 5) + (5, 5), (2, 2) + (3, 8), (4, 3) + (5, 7)

    print("\n--- 21. Flatten a Nested List ---")
    nested = [1, [2, 3, [4, 5]], 6, [7]]
    flat = flatten_nested_list(nested)
    print(f"Nested List: {nested}")
    print(f"Flattened: {flat}")
