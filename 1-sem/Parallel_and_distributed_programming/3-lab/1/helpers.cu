#include <stdio.h>
#include "helpers.h"

void checkError(cudaError_t cudaStatus, char* msg, int* dev_a, int* dev_c) {
	if (cudaStatus == cudaSuccess)
		return;

	printf(msg);
	cudaFree(dev_a);
	cudaFree(dev_c);
}

void printArray(int* a, int size) {
	for (int i = 0; i < size; i++)
		printf("%d ", a[i]);
}

void printMatrix(int* matrix, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%d ", matrix[i * size + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void test_min(int result, int* array, int arraySize) {
	int v = array[0];
	for (size_t i = 0; i < arraySize; i++) {
		if (array[i] < v) {
			v = array[i];
		}
	}
	if (result != v) {
		printf("\nError: Wrong answer. Real answear %d\n", v);
	}
}

void test_histogram(int* result, int* array, int arraySize, bool trace) {
	const int PIXELS = 256;
	int* test = new int[PIXELS];
	for (int i = 0; i < PIXELS; i++) {
		test[i] = 0;
	}
	for (int i = 0; i < arraySize; i++) {
		test[array[i]]++;
	}
	int i = 0;
	for (int i = 0; i < PIXELS; i++) {
		if (result[i] != test[i]) {
			printf("\nError: Wrong answer (%d: %d != %d)", i, test[i], result[i]);
			break;
		}
		if (trace) {
			printf("\n %d %d ", test[i], result[i]);
		}
		i++;
	}
}