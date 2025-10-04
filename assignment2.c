#include <stdio.h>
#include <stdlib.h> // Required for EXIT_SUCCESS

// Function to perform Linear Search and count occurrences
// It takes the array, its size, and the target number as input.
void linear_search_with_count(int arr[], int size, int target) {
    int count = 0;
    
    printf("Searching for the number %d...\n", target);
    printf("Positions found (1-based index): ");

    // Iterate through the entire array
    for (int i = 0; i < size; i++) {
        // Check if the current element matches the target
        if (arr[i] == target) {
            count++;
            // Print the position (using 1-based indexing for user readability)
            printf("%d", i + 1);

            // Add a comma if it's not the last found position (a bit complex to check)
            // A simpler way is to just print a space after the position.
            printf(" "); 
        }
    }

    printf("\n--- Search Summary ---\n");

    // Check the results and print the final output
    if (count > 0) {
        printf("SUCCESS: The number %d was found %d time(s).\n", target, count);
    } else {
        printf("FAILURE: The number %d was not found in the array.\n", target);
    }
}

int main() {
    int size;
    int target;

    // 1. Get Array Size
    printf("Enter the size of the array (max 100): ");
    // Ensure that the user enters a positive size
    if (scanf("%d", &size) != 1 || size <= 0 || size > 100) {
        printf("Invalid array size. Exiting.\n");
        return EXIT_FAILURE;
    }

    // Declare the array dynamically or use a fixed size (here fixed for simplicity)
    int arr[100];

    // 2. Get Array Elements
    printf("Enter %d elements for the array:\n", size);
    for (int i = 0; i < size; i++) {
        printf("Element [%d]: ", i + 1);
        if (scanf("%d", &arr[i]) != 1) {
            printf("Invalid input. Exiting.\n");
            return EXIT_FAILURE;
        }
    }

    // 3. Get Target Number
    printf("\nEnter the number you want to search for: ");
    if (scanf("%d", &target) != 1) {
        printf("Invalid input. Exiting.\n");
        return EXIT_FAILURE;
    }

    // 4. Call the search function
    linear_search_with_count(arr, size, target);

    return EXIT_SUCCESS;
}
