// 1. Linear Search with Count
#include <stdio.h>
#include <stdlib.h> // For malloc (though not strictly needed for fixed size)

// Function to perform linear search and count occurrences
void linear_search_with_count(int arr[], int n, int target) {
    int count = 0;
    // We can't know how many times it will occur, so we'll just print the indices
    printf("Positions where %d appears (1-based indexing):\n", target);

    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            count++;
            printf("  Found at index %d (position %d)\n", i, i + 1);
        }
    }

    printf("\nTotal occurrences of %d: %d\n", target, count);
}

int main() {
    int arr[] = {10, 20, 30, 10, 50, 20, 10, 60};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target;

    printf("Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    printf("Enter a number to search: ");
    if (scanf("%d", &target) != 1) {
        printf("Invalid input.\n");
        return 1;
    }

    linear_search_with_count(arr, n, target);

    return 0;
}

//2. Binary Search (Iterative)
#include <stdio.h>

// Function to perform iterative binary search
int iterative_binary_search(int arr[], int n, int target) {
    int low = 0;
    int high = n - 1;
    int mid;
    int steps = 0;

    printf("\nStarting Binary Search for %d...\n", target);

    while (low <= high) {
        steps++;
        // Calculate mid, protecting against potential overflow (though less common with 'int')
        mid = low + (high - low) / 2;

        printf("--- Step %d ---\n", steps);
        printf("Low: %d, High: %d, Mid: %d (Value: %d)\n", 
               low, high, mid, arr[mid]);

        // Check if target is present at mid
        if (arr[mid] == target) {
            printf("Target %d found at index %d in %d steps!\n", target, mid, steps);
            return mid;
        }

        // If target is greater, ignore the left half
        if (arr[mid] < target) {
            low = mid + 1;
        } 
        // If target is smaller, ignore the right half
        else {
            high = mid - 1;
        }
    }

    printf("Target %d not found after %d steps.\n", target, steps);
    return -1; // Element not found
}

int main() {
    // Binary Search requires a sorted array
    int sorted_arr[] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    int n = sizeof(sorted_arr) / sizeof(sorted_arr[0]);
    int target;

    printf("Sorted Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", sorted_arr[i]);
    }
    printf("\n");

    printf("Enter a number to search: ");
    if (scanf("%d", &target) != 1) {
        printf("Invalid input.\n");
        return 1;
    }

    iterative_binary_search(sorted_arr, n, target);

    return 0;
}

//3. Jump Search vs Linear Search (Timing)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ARRAY_SIZE 100000 // Large size to make time difference noticeable

// --- Helper Functions ---

// 1. Jump Search Implementation
int jumpSearch(int arr[], int n, int target) {
    int step = sqrt(n); // Calculate block size
    int prev = 0;
    int comparisons = 0;

    // Finding the block where element is present
    while (arr[step] <= target && step < n) {
        comparisons++;
        prev = step;
        step += sqrt(n);
        if (prev >= n) return -1; // Target not found
    }

    // Performing linear search within the block
    while (arr[prev] < target) {
        comparisons++;
        prev++;
        // If we reach the next block or end of array, target is not present
        if (prev == n || prev == step) return -1;
    }

    // Check if element is found
    if (arr[prev] == target) return prev;

    return -1;
}

// 2. Linear Search Implementation
int linearSearch(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

int main() {
    int *arr = (int *)malloc(ARRAY_SIZE * sizeof(int));
    if (arr == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Initialize the large sorted array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = i * 2;
    }

    int target = ARRAY_SIZE * 2 - 2; // Target near the end of the array
    clock_t start, end;
    double cpu_time_used;
    int index;

    printf("Searching for target %d in an array of size %d.\n", target, ARRAY_SIZE);

    // --- Measure Linear Search Time ---
    start = clock();
    index = linearSearch(arr, ARRAY_SIZE, target);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nLinear Search:\n");
    printf("  Index found: %d\n", index);
    printf("  Time taken: %f seconds\n", cpu_time_used);

    // --- Measure Jump Search Time ---
    start = clock();
    index = jumpSearch(arr, ARRAY_SIZE, target);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nJump Search:\n");
    printf("  Index found: %d\n", index);
    printf("  Time taken: %f seconds\n", cpu_time_used);

    printf("\nObservation: Jump Search (O(sqrt(N))) should be faster than Linear Search (O(N)).\n");

    free(arr);
    return 0;
}
// 4. Interpolation vs Binary Search (Steps)

#include <stdio.h>
#include <stdlib.h>

// --- Search Functions (Return steps) ---

// Binary Search implementation counting steps
int binarySearchSteps(int arr[], int size, int target) {
    int low = 0;
    int high = size - 1;
    int steps = 0;

    while (low <= high) {
        steps++;
        int mid = low + (high - low) / 2;

        if (arr[mid] == target) return steps;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return steps; // Returns total steps taken if not found
}

// Interpolation Search implementation counting steps
int interpolationSearchSteps(int arr[], int size, int target) {
    int low = 0;
    int high = size - 1;
    int steps = 0;
    int pos;

    while (low <= high && target >= arr[low] && target <= arr[high]) {
        steps++;

        // Interpolation formula: Predicts the position based on uniform distribution
        pos = low + (((double)(high - low) / (arr[high] - arr[low])) 
        * (target - arr[low]));

        if (arr[pos] == target) return steps;
        if (arr[pos] < target) low = pos + 1;
        else high = pos - 1;
    }

    return steps; // Returns total steps taken if not found
}

int main() {
    // Uniformly spaced array (10, 20, 30, ... 100)
    int uniform_arr[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int size = sizeof(uniform_arr) / sizeof(uniform_arr[0]);
    int target = 90;

    printf("Array: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] (Uniformly Spaced)\n");
    printf("Target to find: %d\n", target);

    // Binary Search
    int bin_steps = binarySearchSteps(uniform_arr, size, target);
    printf("\nBinary Search Steps: %d\n", bin_steps);

    // Interpolation Search
    int inter_steps = interpolationSearchSteps(uniform_arr, size, target);
    printf("Interpolation Search Steps: %d\n", inter_steps);

    printf("\nObservation: Interpolation Search typically requires fewer steps on uniform data.\n");

    return 0;
}

// 5. Exponential Search
#include <stdio.h>
#include <math.h>

// Helper: Standard Iterative Binary Search (adapted for a specific range)
int binarySearchRange(int arr[], int low, int high, int target) {
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}

// Main Exponential Search function
int exponentialSearch(int arr[], int n, int target) {
    if (n == 0) return -1;

    // Step 1: Check the first element
    if (arr[0] == target) {
        printf("Step 1: Found at index 0.\n");
        return 0;
    }

    // Step 2: Find the range (bound) for the binary search
    int bound = 1;
    printf("\nStep 2: Finding the search range...\n");
    while (bound < n && arr[bound] <= target) {
        printf("  Checking bound %d (Value: %d). Doubling bound.\n", bound, arr[bound]);
        bound *= 2;
    }

    // Step 3: Call binary search on the determined range
    // The range is [bound/2, min(bound, n-1)]
    int range_start = bound / 2;
    int range_end = (bound < n) ? bound : n - 1;

    printf("Step 3: Range found: [%d to %d]. Calling Binary Search...\n", range_start, range_end);

    int result = binarySearchRange(arr, range_start, range_end, target);

    return result;
}

int main() {
    // Large simulated sorted array
    int arr[] = {2, 3, 4, 10, 40, 50, 60, 88, 100, 110, 120, 150, 200, 250, 300};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 110;

    printf("Array size: %d\n", n);
    printf("Target: %d\n", target);

    int index = exponentialSearch(arr, n, target);

    if (index != -1) {
        printf("\nExponential Search: Target %d found at index %d.\n", target, index);
    } else {
        printf("\nExponential Search: Target not found.\n");
    }

    return 0;
}

// 6.Optimized Bubble Sort Comparison
#include <stdio.h>
#include <stdbool.h>

// Function to print the array
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// 1. Standard Bubble Sort (Without optimization)
int standardBubbleSort(int arr[], int n) {
    int i, j, temp;
    int passes = 0;
    
    for (i = 0; i < n - 1; i++) {
        passes++;
        for (j = 0; j < n - i - 1; j++) {
            // Comparison and swap
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return passes;
}

// 2. Optimized Bubble Sort (Stops early if no swap occurs)
int optimizedBubbleSort(int arr[], int n) {
    int i, j, temp;
    int passes = 0;
    bool swapped;

    for (i = 0; i < n - 1; i++) {
        passes++;
        swapped = false;
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true; // A swap occurred in this pass
            }
        }
        
        // If no two elements were swapped by inner loop, then break
        if (swapped == false) {
            printf("\nOPTIMIZATION APPLIED: No swaps in Pass %d. Stopping early.\n", passes);
            break;
        }
    }
    return passes;
}

int main() {
    int arr1[] = {5, 1, 4, 2, 8};
    int n1 = sizeof(arr1) / sizeof(arr1[0]);
    int arr2[] = {5, 1, 4, 2, 8}; // Using the same array data for fair comparison
    int n2 = sizeof(arr2) / sizeof(arr2[0]);

    printf("--- Standard Bubble Sort ---\n");
    printf("Initial Array: ");
    printArray(arr1, n1);

    int passes_std = standardBubbleSort(arr1, n1);

    printf("Sorted Array:  ");
    printArray(arr1, n1);
    printf("Total Passes (Standard): %d\n\n", passes_std);

    printf("--- Optimized Bubble Sort ---\n");
    printf("Initial Array: ");
    printArray(arr2, n2);

    int passes_opt = optimizedBubbleSort(arr2, n2);

    printf("Sorted Array:  ");
    printArray(arr2, n2);
    printf("Total Passes (Optimized): %d\n", passes_opt);
    
    printf("\nExplanation:\n");
    printf("The standard version always runs the outer loop (N-1 times).\n");
    printf("The optimized version uses a 'swapped' flag. If a pass completes without any swaps, it means the array is already sorted, and the algorithm terminates immediately.\n");
    printf("This significantly reduces time complexity from O(N^2) to O(N) in the best-case (already sorted) and for partially sorted arrays.\n");


    // Test with an almost sorted array for clear difference
    int arr_almost_sorted[] = {1, 2, 3, 5, 4, 6, 7};
    int n_almost = sizeof(arr_almost_sorted) / sizeof(arr_almost_sorted[0]);
    int arr_almost_sorted_copy[] = {1, 2, 3, 5, 4, 6, 7};

    printf("\n--- Comparison on ALMOST Sorted Array {1, 2, 3, 5, 4, 6, 7} ---\n");

    passes_std = standardBubbleSort(arr_almost_sorted, n_almost);
    printf("Standard Passes: %d\n", passes_std);

    passes_opt = optimizedBubbleSort(arr_almost_sorted_copy, n_almost);
    printf("Optimized Passes: %d\n", passes_opt);

    return 0;
}
// 7.


