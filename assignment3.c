#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- Problem 1 ---
// Write a C program to define a struct for employee details and display the data of 3 employees.
struct Employee {
    int id;
    char name[50];
    float salary;
};

void problem1() {
    struct Employee emp1 = {101, "Alice", 50000.00};
    struct Employee emp2 = {102, "Bob", 60000.50};
    struct Employee emp3 = {103, "Charlie", 75000.75};
    
    printf("--- Problem 1: Employee Details ---\n");
    printf("Employee 1: ID=%d, Name=%s, Salary=%.2f\n", emp1.id, emp1.name, emp1.salary);
    printf("Employee 2: ID=%d, Name=%s, Salary=%.2f\n", emp2.id, emp2.name, emp2.salary);
    printf("Employee 3: ID=%d, Name=%s, Salary=%.2f\n\n", emp3.id, emp3.name, emp3.salary);
}

// --- Problem 2 ---
// Create a union that stores either an integer or a float. Write a program to demonstrate how to store another value after storing the first.
union Data {
    int i;
    float f;
};

void problem2() {
    union Data data;
    printf("--- Problem 2: Union Demonstration ---\n");
    
    data.i = 10;
    printf("Integer value stored: %d\n", data.i);
    
    data.f = 220.5;
    printf("Float value stored: %.2f\n", data.f);
    printf("After storing float, integer value is garbled: %d\n\n", data.i);
}

// --- Problem 3 ---
// Write a program using a pointer to swap two integer values.
void swapPointers(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void problem3() {
    int x = 10, y = 20;
    printf("--- Problem 3: Swap using Pointers ---\n");
    printf("Before swap: x = %d, y = %d\n", x, y);
    swapPointers(&x, &y);
    printf("After swap: x = %d, y = %d\n\n", x, y);
}

// --- Problem 4 ---
// Write a program using a pointer to access and display a dynamically allocate 2D array.
void problem4() {
    int rows = 3, cols = 4;
    int *arr = (int *)malloc(rows * cols * sizeof(int));
    if (arr == NULL) {
        printf("Memory allocation failed!\n");
        return;
    }
    
    // Initialize the 2D array using pointer arithmetic
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            *(arr + i * cols + j) = i * cols + j + 1;
        }
    }
    
    printf("--- Problem 4: Dynamic 2D Array using Pointer ---\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", *(arr + i * cols + j));
        }
        printf("\n");
    }
    printf("\n");
    
    free(arr);
}

// --- Problem 5 ---
// Write a function that takes two integers and swaps them.
// a. Pass by value: explain why the swap fails.
// b. Pass by reference: explain why the swap is successful.
void swap_by_value(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
}

void swap_by_reference(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void problem5() {
    int val1 = 10, val2 = 20;
    
    printf("--- Problem 5: Pass by Value vs Pass by Reference ---\n");
    
    // Pass by value
    printf("Before pass by value: val1 = %d, val2 = %d\n", val1, val2);
    swap_by_value(val1, val2);
    printf("After pass by value: val1 = %d, val2 = %d\n", val1, val2);
    printf("Explanation: In pass by value, a copy of the variables is passed to the function. The swap happens on these copies, not the original variables. Therefore, the original values remain unchanged.\n\n");
    
    // Pass by reference
    printf("Before pass by reference: val1 = %d, val2 = %d\n", val1, val2);
    swap_by_reference(&val1, &val2);
    printf("After pass by reference: val1 = %d, val2 = %d\n", val1, val2);
    printf("Explanation: In pass by reference, the addresses of the variables are passed. The function can directly access and modify the original variables using these addresses (pointers), making the swap successful.\n\n");
}

// --- Problem 6 ---
// Simulate nested function behavior in C++ using helper functions and function pointers.
// NOTE: C does not have true nested functions like some other languages. This is a simulation using function pointers.
void innerFunction(int x) {
    printf("This is the inner function, x = %d\n", x);
}

void outerFunction(void (*func_ptr)(int)) {
    printf("This is the outer function.\n");
    func_ptr(100);
}

void problem6() {
    printf("--- Problem 6: Simulate Nested Functions ---\n");
    outerFunction(&innerFunction);
    printf("\n");
}

// --- Problem 7 ---
// Define a pointer that contains a pointer to a variable. Demonstrate the change.
void problem7() {
    int a = 10;
    int *ptr_a = &a;
    int **ptr_ptr_a = &ptr_a;
    
    printf("--- Problem 7: Pointer to a Pointer ---\n");
    printf("Value of a: %d\n", a);
    printf("Value of *ptr_a: %d\n", *ptr_a);
    printf("Value of **ptr_ptr_a: %d\n", **ptr_ptr_a);
    
    // Change the value through the double pointer
    **ptr_ptr_a = 50;
    
    printf("\nAfter changing the value via double pointer:\n");
    printf("New value of a: %d\n\n", a);
}

// --- Problem 8 ---
// Pass a pointer to a function that modifies the value of a variable. Demonstrate the change.
void modifyValue(int *ptr) {
    *ptr = *ptr + 10;
}

void problem8() {
    int num = 5;
    printf("--- Problem 8: Modify Value via Pointer ---\n");
    printf("Original value: %d\n", num);
    modifyValue(&num);
    printf("New value after modification: %d\n\n", num);
}

// --- Problem 9 ---
// Write a program to use nested structures.
struct Address {
    int house_no;
    char street[50];
    char city[50];
};

struct Person {
    char name[50];
    int age;
    struct Address addr;
};

void problem9() {
    struct Person p1;
    strcpy(p1.name, "John Doe");
    p1.age = 30;
    p1.addr.house_no = 123;
    strcpy(p1.addr.street, "Main Street");
    strcpy(p1.addr.city, "Anytown");
    
    printf("--- Problem 9: Nested Structures ---\n");
    printf("Person Name: %s\n", p1.name);
    printf("Age: %d\n", p1.age);
    printf("Address: House No. %d, %s, %s\n\n", p1.addr.house_no, p1.addr.street, p1.addr.city);
}

// --- Problem 10 ---
// Write a program that passes a struct pointer to a function and modifies the contents of an array inside the struct.
struct Vector {
    int arr[5];
    int size;
};

void modifyVector(struct Vector *v) {
    for (int i = 0; i < v->size; i++) {
        v->arr[i] = v->arr[i] * 2;
    }
}

void problem10() {
    struct Vector myVector = {{1, 2, 3, 4, 5}, 5};
    
    printf("--- Problem 10: Struct Pointer with Array ---\n");
    printf("Original array: ");
    for (int i = 0; i < myVector.size; i++) {
        printf("%d ", myVector.arr[i]);
    }
    printf("\n");
    
    modifyVector(&myVector);
    
    printf("Modified array: ");
    for (int i = 0; i < myVector.size; i++) {
        printf("%d ", myVector.arr[i]);
    }
    printf("\n\n");
}

// --- Problem 11 ---
// Define a struct for a book (title, author, price) and dynamically allocate memory for 5 books.
// Write a program to input and display the names of all the books.
struct Book {
    char title[100];
    char author[50];
    float price;
};

void problem11() {
    int num_books = 5;
    struct Book *books = (struct Book *)malloc(num_books * sizeof(struct Book));
    if (books == NULL) {
        printf("Memory allocation failed!\n");
        return;
    }
    
    printf("--- Problem 11: Dynamic Struct Array ---\n");
    for (int i = 0; i < num_books; i++) {
        printf("Enter details for Book %d:\n", i + 1);
        printf("Title: ");
        scanf(" %[^\n]s", (books + i)->title);
        printf("Author: ");
        scanf(" %[^\n]s", (books + i)->author);
        printf("Price: ");
        scanf("%f", &(books + i)->price);
    }
    
    printf("\nDisplaying all book titles:\n");
    for (int i = 0; i < num_books; i++) {
        printf("Book %d Title: %s\n", i + 1, (books + i)->title);
    }
    printf("\n");
    
    free(books);
}

// --- Problem 12 ---
// Write a program that contains a structure with a pointer to an integer. Write a program to access the value using the pointer.
struct PtrHolder {
    int *ptr;
};

void problem12() {
    int value = 42;
    struct PtrHolder p;
    p.ptr = &value;
    
    printf("--- Problem 12: Struct with Pointer ---\n");
    printf("Value accessed via struct pointer: %d\n\n", *p.ptr);
}

// --- Problem 13 ---
// Create a program to allocate memory dynamically to store elements of an array.
void problem13() {
    int n, i;
    int *ptr;
    
    printf("--- Problem 13: Dynamic Array Allocation ---\n");
    printf("Enter number of elements: ");
    scanf("%d", &n);
    
    ptr = (int *)malloc(n * sizeof(int));
    
    if (ptr == NULL) {
        printf("Memory allocation failed!\n");
        return;
    }
    
    printf("Enter elements:\n");
    for (i = 0; i < n; i++) {
        scanf("%d", ptr + i);
    }
    
    printf("Elements are: ");
    for (i = 0; i < n; i++) {
        printf("%d ", *(ptr + i));
    }
    printf("\n\n");
    
    free(ptr);
}

// --- Problem 14 ---
// Create an array of pointers to strings. Write a program to input and display names using this array.
void problem14() {
    char *names[3];
    char buffer[50];
    
    printf("--- Problem 14: Array of Pointers to Strings ---\n");
    
    for (int i = 0; i < 3; i++) {
        printf("Enter name %d: ", i + 1);
        scanf(" %[^\n]s", buffer);
        names[i] = (char *)malloc(strlen(buffer) + 1);
        if (names[i] == NULL) {
            printf("Memory allocation failed!\n");
            return;
        }
        strcpy(names[i], buffer);
    }
    
    printf("\nDisplaying names:\n");
    for (int i = 0; i < 3; i++) {
        printf("Name %d: %s\n", i + 1, names[i]);
    }
    printf("\n");
    
    for (int i = 0; i < 3; i++) {
        free(names[i]);
    }
}

// --- Problem 15 ---
// Define a struct for a car (model, year, price). Use a pointer to access and modify the values of the struct.
struct Car {
    char model[50];
    int year;
    float price;
};

void problem15() {
    struct Car c1 = {"Tesla Model S", 2022, 90000.00};
    struct Car *ptr_c1 = &c1;
    
    printf("--- Problem 15: Struct Pointer to Car ---\n");
    printf("Original Car Details:\n");
    printf("Model: %s, Year: %d, Price: %.2f\n", ptr_c1->model, ptr_c1->year, ptr_c1->price);
    
    // Modify values using the pointer
    ptr_c1->price = 95000.00;
    
    printf("Modified Car Details:\n");
    printf("Model: %s, Year: %d, Price: %.2f\n\n", c1.model, c1.year, c1.price);
}

// --- Problem 16 ---
// Write a struct that contains a union as one of its members. Write a program to demonstrate how to access and modify the values.
struct DataContainer {
    int type; // 0 for int, 1 for float
    union {
        int i_val;
        float f_val;
    } value;
};

void problem16() {
    struct DataContainer d;
    
    printf("--- Problem 16: Struct with Union Member ---\n");
    
    d.type = 0;
    d.value.i_val = 150;
    printf("Stored integer value: %d\n", d.value.i_val);
    
    d.type = 1;
    d.value.f_val = 3.14159;
    printf("Stored float value: %.4f\n\n", d.value.f_val);
}

// --- Problem 17 ---
// Create a function pointer that points to a function doing different arithmetic operations (add, subtract, multiply) based on user input.
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

void problem17() {
    int choice, x, y, result;
    int (*op_ptr)(int, int);
    
    printf("--- Problem 17: Function Pointer for Arithmetic ---\n");
    printf("Enter two numbers: ");
    scanf("%d %d", &x, &y);
    printf("Select operation:\n1. Add\n2. Subtract\n3. Multiply\n");
    printf("Enter choice: ");
    scanf("%d", &choice);
    
    switch (choice) {
        case 1:
            op_ptr = add;
            break;
        case 2:
            op_ptr = subtract;
            break;
        case 3:
            op_ptr = multiply;
            break;
        default:
            printf("Invalid choice!\n\n");
            return;
    }
    
    result = op_ptr(x, y);
    printf("Result: %d\n\n", result);
}

// --- Problem 18 ---
// Write a program to dynamically allocate an array of integers and returns the sum of all elements.
int sumDynamicArray() {
    int n, i, sum = 0;
    int *arr;
    
    printf("--- Problem 18: Sum of Dynamic Array ---\n");
    printf("Enter number of elements: ");
    scanf("%d", &n);
    
    arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) {
        printf("Memory allocation failed!\n\n");
        return -1;
    }
    
    printf("Enter elements:\n");
    for (i = 0; i < n; i++) {
        scanf("%d", arr + i);
    }
    
    for (i = 0; i < n; i++) {
        sum += *(arr + i);
    }
    
    free(arr);
    return sum;
}

void problem18_wrapper() {
    int total = sumDynamicArray();
    if (total != -1) {
        printf("Sum of elements: %d\n\n", total);
    }
}

// --- Problem 19 ---
// Write a recursive function that calculates the factorial of a number using a pointer to store the result.
void factorialRecursive(int n, unsigned long long *result) {
    if (n == 0) {
        *result = 1;
        return;
    }
    unsigned long long temp;
    factorialRecursive(n - 1, &temp);
    *result = (unsigned long long)n * temp;
}

void problem19() {
    int num;
    unsigned long long fact = 1;
    printf("--- Problem 19: Factorial using Recursion and Pointer ---\n");
    printf("Enter a non-negative integer: ");
    scanf("%d", &num);
    if (num < 0) {
        printf("Factorial is not defined for negative numbers.\n\n");
        return;
    }
    factorialRecursive(num, &fact);
    printf("Factorial of %d is %llu\n\n", num, fact);
}

// --- Problem 20 ---
// Sort an array of strings in descending order of marks.
// NOTE: The question says "marks" but provides no marks in the context.
// Assuming "sort an array of strings in descending order of marks" means sorting based on some score.
// I will sort strings in descending order of their length for demonstration.
void problem20() {
    printf("--- Problem 20: Sort Strings in Descending Order of Length ---\n");
    printf("This program will sort strings based on their length, as 'marks' are not provided.\n\n");

    char *strings[] = {"apple", "banana", "kiwi", "grapefruit", "orange"};
    int n = sizeof(strings) / sizeof(strings[0]);
    
    // Bubble sort to sort based on length in descending order
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (strlen(strings[j]) < strlen(strings[j+1])) {
                char *temp = strings[j];
                strings[j] = strings[j+1];
                strings[j+1] = temp;
            }
        }
    }
    
    printf("Sorted strings (descending order of length):\n");
    for (int i = 0; i < n; i++) {
        printf("%s\n", strings[i]);
    }
    printf("\n");
}

int main() {
    int choice;
    do {
        printf("Select a problem to run (1-20, 0 to exit):\n");
        printf("1. Struct for Employees\n");
        printf("2. Union Demonstration\n");
        printf("3. Swap using Pointers\n");
        printf("4. Dynamic 2D Array\n");
        printf("5. Pass by Value vs Pass by Reference\n");
        printf("6. Simulate Nested Functions\n");
        printf("7. Pointer to a Pointer\n");
        printf("8. Modify Value via Pointer\n");
        printf("9. Nested Structures\n");
        printf("10. Struct Pointer with Array\n");
        printf("11. Dynamic Struct Array\n");
        printf("12. Struct with Pointer\n");
        printf("13. Dynamic Array Allocation\n");
        printf("14. Array of Pointers to Strings\n");
        printf("15. Struct Pointer to Car\n");
        printf("16. Struct with Union Member\n");
        printf("17. Function Pointer for Arithmetic\n");
        printf("18. Sum of Dynamic Array\n");
        printf("19. Factorial using Recursion and Pointer\n");
        printf("20. Sort Strings by Length\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1: problem1(); break;
            case 2: problem2(); break;
            case 3: problem3(); break;
            case 4: problem4(); break;
            case 5: problem5(); break;
            case 6: problem6(); break;
            case 7: problem7(); break;
            case 8: problem8(); break;
            case 9: problem9(); break;
            case 10: problem10(); break;
            case 11: problem11(); break;
            case 12: problem12(); break;
            case 13: problem13(); break;
            case 14: problem14(); break;
            case 15: problem15(); break;
            case 16: problem16(); break;
            case 17: problem17(); break;
            case 18: problem18_wrapper(); break;
            case 19: problem19(); break;
            case 20: problem20(); break;
            case 0: printf("Exiting program.\n"); break;
            default: printf("Invalid choice. Please try again.\n");
        }
    } while (choice != 0);

    return 0;
}
