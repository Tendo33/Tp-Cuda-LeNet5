#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<algorithm>



//Cr¨¦ation d'une matrice sur CPU
void MatrixInit(float* M, int n, int p) {
	for (int i = 0; i < n * p; i++) {
		M[i] = float(2.0 * rand() / RAND_MAX - 1);

	}
}

//Affichage d'une matrice sur CPU
void MatrixPrint(float* M, int n, int p) {
	float* im = M;
	printf("Matric shape is:(%d,%d)\n", n, p);

	for (int iy = 0; iy < p; iy++) {
		for (int ix = 0; ix < n; ix++) {
			printf("%f  ", im[ix]);
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

__global__ void cudaMatrixNul(float* M1, float* M2, float* Mout, int n) {
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

int main() {
	const int n = 1500;
	const int p = 1500;
	int nxy = n * p;
	int nBytes = nxy * sizeof(float);

	float* M1;
	float* M2;
	float* Mout;
	float* Mout2;

	M1 = (float*)malloc(nBytes);
	M2 = (float*)malloc(nBytes);
	Mout = (float*)malloc(nBytes);
	Mout2 = (float*)malloc(nBytes);


	MatrixInit(M1, n, p);
	MatrixInit(M2, n, p);
	MatrixInit(Mout, n, p);

	//MatrixPrint(M1, n, p);

	//start count the time off fonction Add on CPU;
	clock_t cpuStart = clock();
	MatrixAdd(M1, M2, Mout, n, p);
	clock_t cpuEnd = clock();
	float cpuTime = (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;// / CLOCKS_PER_SEC
	//MatrixPrint(Mout, n, p);
	printf("ADD fonction CPU time is: %f\n", cpuTime);

	//start count the time off fonction Add on CPU;
	clock_t cpuStart2 = clock();
	MatrixMult(M1, M2, Mout, n);
	clock_t cpuEnd2 = clock();
	float cpuTime2 = (float)(cpuEnd2 - cpuStart2) / CLOCKS_PER_SEC;// / CLOCKS_PER_SEC
	//MatrixPrint(Mout, n, p);
	printf("Multi fonction CPU time is: %f\n", cpuTime2);



	//Allocate GPU memory
	float* d_a, * d_b, * d_c;
	cudaMalloc((void**)&d_a, nBytes);
	cudaMalloc((void**)&d_b, nBytes);
	cudaMalloc((void**)&d_c, nBytes);

	//Initialize grid and block size
	dim3 block(128, 1);
	dim3 grid((n + block.x - 1) / block.x, (p + block.y - 1) / block.y);

	//Copy data from cpu to gpu
	cudaMemcpy(d_a, M1, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, M2, nBytes, cudaMemcpyHostToDevice);

	//start count the time off fonction Add on GPU;
	clock_t gpuStart = clock();
	cudaMatrixAdd << < grid, block >> > (d_a, d_b, d_c, n, p);
	cudaDeviceSynchronize();
	clock_t gpuEnd = clock();
	float gpuTime = (float)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;// / CLOCKS_PER_SEC;

	//The result is copied from the gpu back to the cpu
	cudaMemcpy(Mout2, d_c, nBytes, cudaMemcpyDeviceToHost);
	//MatrixPrint(Mout2, n, p);
	printf("ADD fonction GPU time is:%f\n", gpuTime);


	clock_t gpuStart2 = clock();
	cudaMatrixNul << < grid, block >> > (d_a, d_b, d_c, n);
	cudaDeviceSynchronize();
	clock_t gpuEnd2 = clock();
	float gpuTime2 = (float)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;// / CLOCKS_PER_SEC;

	//The result is copied from the gpu back to the cpu
	cudaMemcpy(Mout2, d_c, nBytes, cudaMemcpyDeviceToHost);
	//MatrixPrint(Mout2, n, p);
	printf("Multi fonction GPU time is:%f\n", gpuTime2);



	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(M1);
	free(M2);
	free(Mout);
	free(Mout2);

}