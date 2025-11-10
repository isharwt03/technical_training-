# Problem Statement: Lab Sheet-04 Technical Training
# Objective: To understand the basic concept of programming.

import math
import collections
import itertools
import datetime
import re

# ==============================================================================
# 1. BASICS OF PYTHON (Tasks 1-3)
# ==============================================================================
print("="*50)
print("1. BASICS OF PYTHON")
print("="*50)

# 1. Print "Hello, World!" using Python.
print("1. Hello, World!")

# 2. Use the print() function to display multiple values on the same line.
print("2. Multiple values:", "Apple", 15, True, sep=" | ")

# 3. Perform basic arithmetic operations (addition, subtraction, multiplication, division, modulus, exponentiation).
a = 15
b = 4
print("\n3. Arithmetic Operations:")
print(f"   Addition (a + b): {a + b}")         # 19
print(f"   Subtraction (a - b): {a - b}")      # 11
print(f"   Multiplication (a * b): {a * b}")   # 60
print(f"   Division (a / b): {a / b}")         # 3.75 (Float)
print(f"   Modulus (a % b): {a % b}")          # 3 (Remainder)
print(f"   Exponentiation (a ** b): {a ** b}") # 15^4 = 50625

# ==============================================================================
# 2. CONDITIONAL STATEMENTS (Tasks 4-6)
# ==============================================================================
print("\n"+"="*50)
print("2. CONDITIONAL STATEMENTS")
print("="*50)

# 4. Write a program to check if a number is odd or even.
num_check = 13
if num_check % 2 == 0:
    print(f"4. The number {num_check} is Even.")
else:
    print(f"4. The number {num_check} is Odd.")

# 5. Write a self-wise categorize a number as positive, negative, or zero.
num_sign = -5
if num_sign > 0:
    print(f"5. The number {num_sign} is Positive.")
elif num_sign < 0:
    print(f"5. The number {num_sign} is Negative.")
else:
    print(f"5. The number {num_sign} is Zero.")

# 6. Determine if a year is a leap year (Divisible by 4, but not by 100 unless divisible by 400).
year = 2024
is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
print(f"6. Is {year} a leap year? {is_leap}") # True

# ==============================================================================
# 3. LOOPS (Tasks 7-9)
# ==============================================================================
print("\n"+"="*50)
print("3. LOOPS")
print("="*50)

# 7. Print numbers from 1 to N using a for loop.
N_for = 5
print(f"7. Numbers from 1 to {N_for} (for loop):", end=" ")
for i in range(1, N_for + 1):
    print(i, end=" ")
print()

# 8. Print the square of numbers from 0 to N-1.
N_sq = 4
print(f"8. Squares from 0 to {N_sq-1}:", end=" ")
for i in range(N_sq):
    print(f"{i*i}", end=" ")
print()

# 9. Use a while loop to reverse a number.
num_to_reverse = 12345
original_num = num_to_reverse
reversed_num = 0

while num_to_reverse > 0:
    digit = num_to_reverse % 10          # Get the last digit
    reversed_num = reversed_num * 10 + digit # Add the digit to the reversed number
    num_to_reverse //= 10                # Remove the last digit

print(f"9. Reversed number of {original_num}: {reversed_num}")

# ==============================================================================
# 4. FUNCTIONS (Tasks 10-12)
# ==============================================================================
print("\n"+"="*50)
print("4. FUNCTIONS")
print("="*50)

# 10. Write a function to return the factorial of a number.
def calculate_factorial(n):
    """Calculates the factorial of a non-negative integer n."""
    if n < 0:
        return "Factorial is not defined for negative numbers"
    # math.factorial is the standard way, but we can implement it with a loop too:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(f"10. Factorial of 5: {calculate_factorial(5)}")

# 11. Create a function to check whether a number is prime.
def is_prime(n):
    """Checks if a positive integer n is a prime number."""
    if n <= 1:
        return False
    # Check for factors from 2 up to the square root of n
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

print(f"11. Is 17 prime? {is_prime(17)}")
print(f"11. Is 15 prime? {is_prime(15)}")

# 12. Write a function to return the Fibonacci sequence up to N terms.
def generate_fibonacci(n):
    """Generates the Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    if n == 1:
        return [0]

    sequence = [0, 1]
    # Generate the remaining n-2 terms
    while len(sequence) < n:
        next_term = sequence[-1] + sequence[-2]
        sequence.append(next_term)
    return sequence

print(f"12. Fibonacci sequence (8 terms): {generate_fibonacci(8)}")

# ==============================================================================
# 5. DATA STRUCTURES (Tasks 13-17)
# ==============================================================================
print("\n"+"="*50)
print("5. DATA STRUCTURES")
print("="*50)

# 13. Create a list and perform append, insert, remove, and sort operations.
my_list = [50, 10, 40, 20]
print(f"13. Initial list: {my_list}")

# Append: adds an element to the end
my_list.append(60)
print(f"    After append(60): {my_list}")

# Insert: adds an element at a specific index (index 1)
my_list.insert(1, 15)
print(f"    After insert(1, 15): {my_list}")

# Remove: removes the first occurrence of a value
my_list.remove(40)
print(f"    After remove(40): {my_list}")

# Sort: sorts the list in place
my_list.sort()
print(f"    After sort(): {my_list}")

# 14. Use list comprehensions to generate a list of squares of even numbers.
# We'll generate squares of even numbers from 1 to 10
squares_of_evens = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"14. Squares of evens (1 to 10): {squares_of_evens}")

# 15. Create a tuple and demonstrate immutability.
my_tuple = (1, 2, 'a', 'b')
print(f"15. Tuple created: {my_tuple}")

# Attempting to change an element will result in a TypeError (demonstrating immutability)
try:
    # my_tuple[0] = 99 # This line would cause an error
    print("    Tuple element cannot be changed (e.g., my_tuple[0] = 99 would fail).")
except TypeError as e:
    print(f"    Error on modification attempt: {e}")

# 16. Use a set to remove duplicates from a list.
list_with_duplicates = [10, 20, 30, 20, 10, 40, 50, 30]
unique_elements = set(list_with_duplicates)
print(f"16. Original list: {list_with_duplicates}")
print(f"    List with duplicates removed (Set): {unique_elements}")

# 17. Create a dictionary and perform key-value operations.
my_dict = {"name": "Alice", "age": 30, "city": "New York"}
print(f"17. Initial dictionary: {my_dict}")

# Add a new key-value pair
my_dict["job"] = "Engineer"
print(f"    After adding 'job': {my_dict}")

# Update a value
my_dict["age"] = 31
print(f"    After updating 'age': {my_dict}")

# Access a value
city = my_dict["city"]
print(f"    Accessed value for 'city': {city}")

# Remove a key-value pair
del my_dict["city"]
print(f"    After deleting 'city': {my_dict}")

# ==============================================================================
# 6. STRINGS (Tasks 18-22)
# ==============================================================================
print("\n"+"="*50)
print("6. STRINGS")
print("="*50)

# 18. Write a program to count vowels in a string.
def count_vowels(s):
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count

test_string_vowel = "Programming is Fun"
print(f"18. Vowel count in '{test_string_vowel}': {count_vowels(test_string_vowel)}")

# 19. Reverse a string using slicing.
test_string_reverse = "Python"
reversed_string = test_string_reverse[::-1]
print(f"19. Original string: '{test_string_reverse}'")
print(f"    Reversed string: '{reversed_string}'")

# 20. Check if a string is a palindrome.
def is_palindrome(s):
    # Remove spaces and convert to lowercase for accurate check
    s = s.replace(" ", "").lower()
    return s == s[::-1]

print(f"20. Is 'madam' a palindrome? {is_palindrome('madam')}")
print(f"20. Is 'Python' a palindrome? {is_palindrome('Python')}")

# 21. Format a string using f-strings.
product = "Laptop"
price = 999.99
formatted_string = f"21. The {product} is currently on sale for ${price:.2f}."
print(formatted_string)

# 22. Formulate a string to generate all combinations of two lists.
# (Interpretation: Create a string output showing the combinations of elements.)
list1_comb = ['A', 'B']
list2_comb = [1, 2, 3]
combination_list = []

for x in list1_comb:
    for y in list2_comb:
        combination_list.append(f"({x}, {y})")

print(f"22. Combinations of {list1_comb} and {list2_comb}: {', '.join(combination_list)}")

# ==============================================================================
# 7. ADVANCED PYTHON (Tasks 23-26)
# ==============================================================================
print("\n"+"="*50)
print("7. ADVANCED PYTHON (Optional for Extra Credit)")
print("="*50)

# 23. Use itertools.permutations() to generate all combinations of two lists.
# Note: The prompt specifically mentions 'permutations' but asks for 'combinations of two lists'.
# itertools.product is typically used for combinations of elements from different lists.
# We will show both product (cross-product) and permutations (order matters within a single list).

# Using itertools.product for cross-product (like the combinations from Q22)
list_A = [1, 2]
list_B = ['x', 'y']
product_result = list(itertools.product(list_A, list_B))
print(f"23. itertools.product (Cross-product of {list_A} and {list_B}): {product_result}")

# Using itertools.permutations for orderings of elements within a single list
list_P = [10, 20, 30]
permutations_result = list(itertools.permutations(list_P, 2)) # Permutations of length 2
print(f"23. itertools.permutations of {list_P} (length 2): {permutations_result}")


# 24. Use collections.Counter() to count character frequency in a string.
test_string_freq = "mississippi"
char_counts = collections.Counter(test_string_freq)
print(f"24. Character frequency in '{test_string_freq}': {char_counts}")
print(f"    Frequency of 's': {char_counts['s']}")

# 25. Parse a date string and format it using datetime.
date_string = "2025-10-07"
# Parse the string into a datetime object
date_object = datetime.datetime.strptime(date_string, "%Y-%m-%d")

# Format the datetime object into a new, friendly string format
formatted_date = date_object.strftime("%B %d, %Y (%A)")

print(f"25. Original date string: {date_string}")
print(f"    Formatted date: {formatted_date}")

# 26. Use regular expressions to validate an email address. (Task 25 in the sheet image)
def validate_email(email):
    # A simple but common regex pattern for basic email validation
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.fullmatch(pattern, email):
        return True
    return False

email_1 = "test.user@example.com"
email_2 = "invalid-email"

print(f"26. Is '{email_1}' valid? {validate_email(email_1)}")
print(f"26. Is '{email_2}' valid? {validate_email(email_2)}")

print("\n"+"="*50)
print("All tasks completed.")
print("="*50)
