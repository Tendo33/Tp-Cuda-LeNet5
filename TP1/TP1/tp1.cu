#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<algorithm>


//Cr¨¦ation d'une matrice sur CPU
void MatrixInit(float* M, int n, int p) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			M[j] = float(rand());
		}
	}
}

//Affichage d'une matrice sur CPU
void MatrixPrint(float* M, int n, int p) {
	float* im = M;
	printf("\nMatric shape is:(%d,%d)\n", n, p);

	for (int iy = 0; iy < p; iy++) {
		for (int ix = 0; ix < n; ix++) {
			printf("Matric%f", im[ix]);
		}
		im += n;

		printf("\n");
	}

}

//Addition de deux matrices sur CPU
void MatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {

	float* ia = M1;
	float* ib = M2;
	float* ic = Mout;
	for (int iy = 0; iy < p; iy++) {
		for (int ix = 0; ix < n; ix++) {
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += n;
		ib += n;
		ic += n;
	}


}

//Addition de deux matrices sur GPU
__global__ void cudaMatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {

	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int idx = iy * n + ix;
	if (ix < n && iy < p) {
		Mout[idx] = M1[idx] + M2[idx];
	}

}

//Multiplication de deux matrices NxN sur CPU
void MatrixMult(float* M1, float* M2, float* Mout, int n) {

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < n; k++)
			{
				float a = M1[i * n + k];
				float b = M2[k * n + j];
				sum += a * b;
			}
			Mout[i * n + j] = sum;
		}
}

//Multiplication de deux matrices NxN sur GPU

__global__ void cudaMatrixAdd(float* M1, float* M2, float* Mout, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	int sum = 0;
	for (int k = 0; k < n; k++) {

		int a = M1[j * n + k];
		int b = M2[k * n + i];
		sum += a * b;
	}
	Mout[j * n + i] = sum;
}

