#include<stdio.h>
#include<conio.h>
int main() {
    // --- Problem 1: String reversal ---
    char str[] = "hello world";
    printf("Original string: '%s'\n", str);
    reverseString(str);
    printf("Reversed string: '%s'\n\n", str);

    // --- Problem 2: Replacing substrings ---
    const char* originalStr = "I love programming. Programming is fun!";
    char* replacedStr = replaceSubstring(originalStr, "Programming", "Coding");
    printf("Original string: '%s'\n", originalStr);
    if (replacedStr) {
        printf("String with substring replaced: '%s'\n\n", replacedStr);
        free(replacedStr);
    } else {
        printf("Memory allocation failed for replacement.\n\n");
    }

    // --- Problem 3: Searching for an element ---
    int arr[] = {10, 20, 30, 40, 50};
    int size = sizeof(arr) / sizeof(arr[0]);
    int target = 30;
    int index = linearSearch(arr, size, target);
    printf("Array: {");
    for(int i = 0; i < size; i++) printf("%d%s", arr[i], (i == size - 1) ? "" : ", ");
    printf("}\n");
    if (index != -1) {
        printf("Found %d at index %d.\n\n", target, index);
    } else {
        printf("%d not found in the array.\n\n", target);
    }

    // --- Problem 4: Sorting an array (Bubble Sort) ---
    int unsortedArr[] = {64, 34, 25, 12, 22, 11, 90};
    int sortSize = sizeof(unsortedArr) / sizeof(unsortedArr[0]);
    
    printf("Original array: {");
    for(int i = 0; i < sortSize; i++) printf("%d%s", unsortedArr[i], (i == sortSize - 1) ? "" : ", ");
    printf("}\n");
    
    bubbleSort(unsortedArr, sortSize, true);
    printf("Sorted ascending: {");
    for(int i = 0; i < sortSize; i++) printf("%d%s", unsortedArr[i], (i == sortSize - 1) ? "" : ", ");
    printf("}\n");

    bubbleSort(unsortedArr, sortSize, false);
    printf("Sorted descending: {");
    for(int i = 0; i < sortSize; i++) printf("%d%s", unsortedArr[i], (i == sortSize - 1) ? "" : ", ");
    printf("}\n\n");
    
    // --- Problem 5: Reversing an array ---
    int arrToReverse[] = {1, 2, 3, 4, 5};
    int reverseSize = sizeof(arrToReverse) / sizeof(arrToReverse[0]);
    printf("Original array: {");
    for(int i = 0; i < reverseSize; i++) printf("%d%s", arrToReverse[i], (i == reverseSize - 1) ? "" : ", ");
    printf("}\n");
    reverseArray(arrToReverse, reverseSize);
    printf("Reversed array: {");
    for(int i = 0; i < reverseSize; i++) printf("%d%s", arrToReverse[i], (i == reverseSize - 1) ? "" : ", ");
    printf("}\n\n");
    
    // --- Problem 6: Removing duplicates ---
    int arrWithDuplicates[] = {1, 2, 2, 3, 4, 4, 5, 5, 1};
    int duplicateSize = sizeof(arrWithDuplicates) / sizeof(arrWithDuplicates[0]);
    printf("Original array with duplicates: {");
    for(int i = 0; i < duplicateSize; i++) printf("%d%s", arrWithDuplicates[i], (i == duplicateSize - 1) ? "" : ", ");
    printf("}\n");
    bubbleSort(arrWithDuplicates, duplicateSize, true); // Sort first for the function to work
    int newSize = removeDuplicates(arrWithDuplicates, duplicateSize);
    printf("Array with duplicates removed: {");
    for(int i = 0; i < newSize; i++) printf("%d%s", arrWithDuplicates[i], (i == newSize - 1) ? "" : ", ");
    printf("}\n\n");
    
    // --- Problem 7: Merging two arrays ---
    int arr1[] = {1, 3, 5, 7};
    int size1 = sizeof(arr1) / sizeof(arr1[0]);
    int arr2[] = {2, 4, 6, 8};
    int size2 = sizeof(arr2) / sizeof(arr2[0]);
    int* mergedArr = mergeSortedArrays(arr1, size1, arr2, size2);
    if (mergedArr) {
        printf("Array 1: {");
        for(int i = 0; i < size1; i++) printf("%d%s", arr1[i], (i == size1 - 1) ? "" : ", ");
        printf("}\n");
        printf("Array 2: {");
        for(int i = 0; i < size2; i++) printf("%d%s", arr2[i], (i == size2 - 1) ? "" : ", ");
        printf("}\n");
        printf("Merged sorted array: {");
        for(int i = 0; i < size1 + size2; i++) printf("%d%s", mergedArr[i], (i == size1 + size2 - 1) ? "" : ", ");
        printf("}\n\n");
        free(mergedArr);
    } else {
        printf("Failed to merge arrays.\n\n");
    }

    // --- File I/O Operations (Problems 8, 9, 10) ---
    const char *sourceFile = "source.txt";
    const char *destinationFile = "destination.txt";

    // Create a source file for demonstration
    FILE *f = fopen(sourceFile, "w");
    if (f) {
        fprintf(f, "This is the first line.\n");
        fprintf(f, "This is the second line with a secret word.\n");
        fprintf(f, "The quick brown fox jumps over the lazy dog.\n");
        fclose(f);
    }

    // Problem 8: Reading from a text file
    readFile(sourceFile);

    // Problem 9: Copying files
    copyFile(sourceFile, destinationFile);

    // Problem 10: Searching within a file
    searchFileForWord(destinationFile, "secret word");
    searchFileForWord(destinationFile, "non-existent");
    
    // Clean up created files
    remove(sourceFile);
    remove(destinationFile);
    printf("\nCleaned up demonstration files: '%s' and '%s'.\n", sourceFile, destinationFile);

    return 0;
}
