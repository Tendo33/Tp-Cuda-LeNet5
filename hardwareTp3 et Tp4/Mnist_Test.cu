#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <time.h>
#include <algorithm>


#define WIDTH 28
#define HEIGHT 28


void charBckgrndPrint(char *str, int rgb[3]){
  printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, int ***img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row][col]);
    }
    printf("\n");
  }
}

void Matrix2DInitRand(float* M, int n, int p)
{
	srand(time(0));
	for (int i = 0; i < n * p; i++) {
		
			M[i] = (float)rand() / (float)RAND_MAX;
		
	}
}

void Matrix3DInitZero(float* M, int fm, int n, int p)
{
	for (int i = 0; i < n*p*fm; i++) {
		M[i] = 0;
	}
}

void Matrix3DInitRand(float* M, int fm, int n, int p)
{
	srand(time(0));
	for (int i = 0; i < n * p * fm; i++) {
		M[i] = (float)rand() / (float)RAND_MAX;
	}
}





void Matrix2DPrint(float* M, int n, int p)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			printf("%f |", M[i * n + j]);
		}
		printf("\n");
	}
}



void activation_softmax(float* vin, int n, float* vout)
{
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum += vin[i];
	}
	for (int i = 0; i < n; i++) {
		vout[i] = vin[i] / sum;
	}
}

__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_ligne, int M_colonne, int kernel_size, int nb_kernel, int Mout_ligne, int Mout_colonne) {

	//Convolution d'une matrice par un kernel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int lig = blockIdx.y * blockDim.y + threadIdx.y;
	int chn = blockIdx.z * blockDim.z + threadIdx.z;

	float s = 0.0;

	if (lig < Mout_ligne && col < Mout_colonne)
	{
		int tot = M_ligne * M_colonne * nb_kernel;

		for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
			for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
				for (int n_k = 0; n_k < nb_kernel; n_k++)
				{
					s += M[(lig + kernel_lig) * M_colonne + col + kernel_col + n_k * tot] * kernel[kernel_lig * kernel_size + kernel_col + n_k * nb_kernel];
				}
			}
		}
		Mout[chn * Mout_ligne * Mout_colonne + lig * Mout_colonne + col] = s;
	}
}

__global__ void cudaMeanPool(float* M, float* Mout, int M_ligne, int M_colonne, int M_prof, int meanpool_size, int Mout_ligne, int Mout_colonne) {
	/*
	  cudaMeanPool réalise un sous-échantillonanage par moyennage 2x2 pixels vers 1 pixel.
	*/

	// On effectuera les calculs sur les lignes et colonnes paires. 
	// ATTENTION: Dans ce cas blockIdx.x et threadIdx.x vont de 0 à 13 donc lig et col de 0 à 26.
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int lig = blockIdx.y * blockDim.y + threadIdx.y;


	float s;
	int tot_meanpool = meanpool_size * meanpool_size;
	int tot_M = M_ligne * M_colonne;
	int tot_Mout = Mout_ligne * Mout_colonne;

	for (int n_prof = 0; n_prof < M_prof; n_prof++) {
		s = 0.0;

		for (int meanpool_lig = 0; meanpool_lig < meanpool_size; meanpool_lig++) {
			for (int meanpool_col = 0; meanpool_col < meanpool_size; meanpool_col++) {
				s += M[(lig + meanpool_lig) * M_colonne + col + meanpool_col + n_prof * tot_M] / tot_meanpool;
			}
		}

		Mout[blockIdx.x * Mout_colonne + threadIdx.x + n_prof * tot_Mout] = s;

	}
}

__global__ void activation_tanh(float* M, int M_ligne, int M_colonne, int M_prof, float* Mout) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int lig = blockIdx.y * blockDim.y + threadIdx.y;

	int tot_M = M_ligne * M_colonne;

	for (int n_prof = 0; n_prof < M_prof; n_prof++) {
		Mout[lig * M_colonne + col + n_prof * tot_M] = tanh(M[lig * M_colonne + col + n_prof * tot_M]);
	}

}

__global__ void Dense(float* A, float* v, float* vout, int col) {
	int lig = blockIdx.y * blockDim.y + threadIdx.y;
	// Handling arbitrary vector size
	for (int k = 0; k < col; k++) {
		vout[lig] += A[lig * col + k] * v[k];
	}
}


int main(int argc, char* argv[]) {
  int i, j;
  int ***img;
  int color[3]={255,0,0};
  unsigned int magic, nbImg, nbRows, nbCols;
  unsigned char val;
  FILE *fptr;

  // Malloc image
  img = (int ***)malloc(HEIGHT*sizeof(int **));
  for(i=0; i<HEIGHT; i++){
    img[i]= (int **)malloc(WIDTH*sizeof(int *));
    for(j=0; j<WIDTH; j++){
      img[i][j] = (int *)malloc(sizeof(int)*3);
    }
  }

  //Open File
  if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
    printf("Can't open file");
    exit(1);
  }

  //Read File
  fread(&magic, sizeof(int), 1, fptr);
  fread(&nbImg, sizeof(int), 1, fptr);
  fread(&nbRows, sizeof(int), 1, fptr);
  fread(&nbCols, sizeof(int), 1, fptr);

  printf("Nb Magic : %u \n", magic);
  printf("Nb Img : %u \n", nbImg);
  printf("Nb Rows : %u \n", nbRows);
  printf("Nb Cols : %u \n", nbCols);

    for(i=0; i<HEIGHT; i++){
    for(j=0; j<WIDTH; j++){ 
      fread(&val, sizeof(unsigned char), 1, fptr);  
      img[i][j][0]=(int)val*color[0]/255;
      img[i][j][1]=(int)val*color[1]/255;
      img[i][j][2]=(int)val*color[2]/255;
    }
  }

  printf("image original:");
  printf("\n\n");
  imgColorPrint(HEIGHT, WIDTH, img);
  printf("******************************************");

	 /*Layer 1
	 Initialisation de la matrice d'entrée*/
	float* raw_data;
	raw_data = (float*)malloc(sizeof(float) * 32 * 32);
	Matrix2DInitRand(raw_data, 32, 32);
	

	/* Initialisation de la matrice de sortie de la première conv*/
	float* C1_data;
	C1_data = (float*)malloc(sizeof(float) * 6 * 28 * 28);
	Matrix3DInitZero(C1_data, 6, 28, 28);

	 //Initialisation de la matrice de sortie du sous-echantillonnage
	float* S1_data;
	S1_data = (float*)malloc(sizeof(float) * 6 * 14 * 14);
	Matrix3DInitZero(S1_data, 6, 14, 14);

	 //Initialisation de la matrice des premiers kernels
	float* C1_kernel;
	C1_kernel = (float*)malloc(sizeof(float) * 6 * 5 * 5);
	Matrix3DInitRand(C1_kernel, 6, 5, 5);

	// Initialisation de la matrice des output layer

	float* S2_data;
	S2_data = (float*)malloc(sizeof(float) * 16 * 5 * 5);
	Matrix3DInitZero(S1_data, 16, 5, 5);
	// Initialisation de la matrice de dense
	float* pred;
	pred = (float*)malloc(sizeof(float) * 10);

	// Copie des matrices dans la mémoire GPU afin d'effectuer les calculs du réseau

	float* d_raw_data, * d_C1_data, * d_C1_data_activated, * d_C1_kernel, * d_S1_data;
	float* d_C2_data, * d_C2_data_activated, * d_C2_kernel, * d_S2_data;
	float* d_dense1_weigths, * d_dense1, * d_dense1_activated;
	float* d_dense2_weigths, * d_dense2, * d_dense2_activated;
	float* d_dense3_weigths, * d_dense3, * d_pred;

	cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
	cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
	cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
	cudaMalloc((void**)&d_C1_data_activated, sizeof(float) * 28 * 28 * 6);
	cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6);
	cudaMalloc((void**)&d_C2_data, sizeof(float) * 16 * 10 * 10);
	cudaMalloc((void**)&d_C2_data_activated, sizeof(float) * 16 * 10 * 10);
	cudaMalloc((void**)&d_C2_kernel, sizeof(float) * 16 * 5 * 5);
	cudaMalloc((void**)&d_S2_data, sizeof(float) * 16 * 5 * 5);

	cudaMalloc((void**)&d_dense1_weigths, sizeof(float) * 120 * 16 * 5 * 5);
	cudaMalloc((void**)&d_dense1, sizeof(float) * 120);
	cudaMalloc((void**)&d_dense1_activated, sizeof(float) * 120);
	cudaMalloc((void**)&d_dense2_weigths, sizeof(float) * 120 * 84);
	cudaMalloc((void**)&d_dense2, sizeof(float) * 84);
	cudaMalloc((void**)&d_dense2_activated, sizeof(float) * 84);
	cudaMalloc((void**)&d_dense3_weigths, sizeof(float) * 84 * 10);
	cudaMalloc((void**)&d_dense3, sizeof(float) * 10);
	cudaMalloc((void**)&d_pred, sizeof(float) * 10);


	dim3 blockDim(32, 32);
	dim3 gridDim(float(28/32), float(28/32));
	 //Layer 2: Première conv2D (6 kernels de taille 5x5 => matrice de taille 6x28x28)
	cudaConv2D << < gridDim, blockDim >> > (d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
	cudaDeviceSynchronize();
	// Activation tanh
	activation_tanh << <28, 28 >> > (d_C1_data, 28, 28, 6, d_C1_data_activated);
	cudaDeviceSynchronize();

	// Layer 3: Sous échantillionnage par moyennage 2x2
	cudaMeanPool << <14, 14 >> > (d_C1_data_activated, d_S1_data, 28, 28, 6, 2, 14, 14);
	cudaDeviceSynchronize();

	// Layer 4: Deuxième conv2D (16 kernels de taille 5x5 => matrice de taille 16x10x10)
	cudaConv2D << <14, 14 >> > (d_S1_data, d_C2_kernel, d_C2_data, 14, 14, 5, 16, 10, 10);
	cudaDeviceSynchronize();
	 //Activation tanh
	activation_tanh << <10, 10 >> > (d_C2_data, 10, 10, 16, d_C2_data_activated);
	cudaDeviceSynchronize();

	// Layer 5: Sous échantillionnage par moyennage 2x2
	cudaMeanPool << <5, 5 >> > (d_C2_data_activated, d_S2_data, 10, 10, 16, 2, 5, 5);
	cudaDeviceSynchronize();

	 //Layer 6: Première couche Dense. size de 120
	Dense << <120, 1 >> > (d_dense1_weigths, d_S2_data, d_dense1, 120);
	cudaDeviceSynchronize();
	// Activation tanh
	activation_tanh << <120, 1 >> > (d_dense1, 120, 1, 1, d_dense1_activated);
	cudaDeviceSynchronize();

	 //Layer 7: Deuxième couche Dense. size 84
	Dense << <84, 1 >> > (d_dense2_weigths, d_dense1, d_dense2, 84);
	cudaDeviceSynchronize();
	 //Activation tanh
	activation_tanh << <84, 1 >> > (d_dense2, 84, 1, 1, d_dense2_activated);
	cudaDeviceSynchronize();

	 //Layer 8: Troisième couche Dense. size 10
	Dense << <10, 1 >> > (d_dense3_weigths, d_dense2, d_dense3, 10);
	cudaDeviceSynchronize();
	 //Activation softmax
	activation_softmax(d_dense3, 10, pred);
	cudaDeviceSynchronize();

	 //Copie du resultat GPU sur CPU
	cudaMemcpy(pred, d_pred, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	
	printf("******************************************");
	printf("Prediction: %f \n", pred);

	cudaFree(d_C1_data);
	cudaFree(d_C1_data_activated);
	cudaFree(d_S1_data);
	cudaFree(d_C1_kernel);
	cudaFree(d_raw_data);

	cudaFree(d_C2_data);
	cudaFree(d_C2_data_activated);
	cudaFree(d_S2_data);
	cudaFree(d_C2_kernel);

	free(C1_data);
	free(S1_data);
	free(raw_data);
	free(C1_kernel);

}