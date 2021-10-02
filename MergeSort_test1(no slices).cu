#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h> 


	long long int N = 1 << 27;
	dim3 dimblock(1024, 1);
	dim3 dimgrid((N+dimblock.x-1)/dimblock.x, 1, 1);
	double iStart = 0, iElaps = 0, iElaps2 = 0, sum_iElaps2 = 0;


double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void checkResult(long long int *hostRef, long long int *gpuRef, long long int N) {
 	double epsilon = 1.0E-8; 
	long long int match = 1;
 	for (long long int i = 0; i < N; i++) {
 		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
 			match = 0;
 			printf("Results do not match!\n");
 			printf("host %lld gpu %lld at current%lld\n", hostRef[i], gpuRef[i], i);
 			//break;
 		}
 	}
 	if (match) 
		printf("Results match.\n"); 
	return;
}


  
//===========================================CPU============================================
void merge(long long int* word,long long int h,long long int t,long long int m,long long int* ans){//合併(想成兩個陣列合併)   
	long long int i=h,j=m+1,k=h;
	while(i<=m && j<=t){
		if(word[i]<=word[j]){//後面比前面大，不用交換
			ans[k]=word[i];
			i++;
		}
		else{//前面比後面大，要交換
			ans[k]=word[j];
			j++;
		}
		k++;
	}
	
	while (i<=m) {
		ans[k]=word[i];
		i++;
		k++;
	}
 
	while (j<=t) {
		ans[k]=word[j];
		j++;
		k++;
	}

	for(long long int i=h;i<=t;i++){
		word[i]=ans[i];
	}
}  
  
void cpu_mergesort(long long int* word,long long int h,long long int t,long long int* ans){//先切(從中間切)   
	if(h<t){  
		long long int m=(h+t)/2;  
          
		cpu_mergesort(word,h,m,ans);//分成兩組再切   
		cpu_mergesort(word,m+1,t,ans);  
          
 		merge(word,h,t,m,ans);//切完再合   
	}  
}  
//==========================================CPU END=============================================

//===========================================GPU============================================





__device__ void gpu_bottomUpMerge(long long int* source, long long int* dest, long long int start, long long int middle, long long int end) {
    long long int i = start;
    long long int j = middle;
    for (long long int k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}



__global__ void gpu_mergesort(long long int* source, long long int* dest, long long int size, long long int width) {

    long long int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    long long int start = width*idx, middle, end;


    if (start < size){
        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
    }
}




void mergesort(long long int *gpu_sorted_data, long long int size, long long int *Ans_gpu_sorted_data) {



    long long int* D_data;
    long long int* D_swp;
    
    cudaMalloc((void**) &D_data, size * sizeof(long long int));
    cudaMalloc((void**) &D_swp, size * sizeof(long long int));

    cudaMemcpy(D_data, gpu_sorted_data, size * sizeof(long long int), cudaMemcpyHostToDevice);
 


    long long int* A = D_data;
    long long int* B = D_swp;

    //long long int nThreads = dimblock.x * dimblock.y * dimblock.z * dimgrid.x * dimgrid.y * dimgrid.z;

    
    for (long long int width = 2; width < (size << 1); width <<= 1) {
	
	iStart = cpuSecond();
        gpu_mergesort<<<dimgrid, dimblock>>>(A, B, size, width);
	iElaps2 = cpuSecond() - iStart;
	sum_iElaps2 += iElaps2;
	printf("width = %lld elapsed time: %7.5f ms\n", width, iElaps2*1000);

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    cudaMemcpy(Ans_gpu_sorted_data, A, size * sizeof(long long int), cudaMemcpyDeviceToHost);
    
    

    cudaFree(A);
    cudaFree(B);
    cudaDeviceReset();
}


//==========================================GPU END=============================================

  
int main(int argc, char **argv)  
{  
	iStart = 0, iElaps = 0, iElaps2 = 0;
	long long int *data, *cpu_sorted_data, *gpu_sorted_data;
	long long int *Ans_cpu_sorted_data, *Ans_gpu_sorted_data;

	if (argc == 2) 
		N = atoi(argv[1]);
 	printf("N = %lld\n", N);

	data = (long long int *)malloc(N*sizeof(long long int));
	cpu_sorted_data = (long long int *)malloc(N*sizeof(long long int));
	gpu_sorted_data = (long long int *)malloc(N*sizeof(long long int));
	Ans_cpu_sorted_data = (long long int *)malloc(N*sizeof(long long int));
	Ans_gpu_sorted_data = (long long int *)malloc(N*sizeof(long long int));
	srand(time(NULL));
	
	for (long long int k=0; k<N; k++)
		data[k] = rand() % (N*10);
	
	printf("===========================CPU====================================\n");

	memcpy(cpu_sorted_data, data, N*sizeof(long long int));

	

	printf("cpu Mergesort is starting...\n");

	iStart = cpuSecond();
	cpu_mergesort(cpu_sorted_data, 0, N-1, Ans_cpu_sorted_data);//cpu_mergesort
	iElaps = cpuSecond() - iStart;
	printf("cpu Mergesort is done.\n");
	printf("Mergesort on CPU elapsed time: %7.5f ms\n", iElaps*1000);

	


	printf("===========================GPU====================================\n");
 
        //printf("//////////////////=====\n"); 

	
	memcpy(gpu_sorted_data, data, N*sizeof(long long int));

	

	printf("GPU Mergesort is starting...\n");
	//iStart = cpuSecond();
	
	sum_iElaps2 = 0;
	mergesort(gpu_sorted_data, N, Ans_gpu_sorted_data);

	//iElaps2 = cpuSecond() - iStart;
	printf("GPU Mergesort is done.\n");
	printf("Mergesort on GPU elapsed time: %7.5f ms\n", sum_iElaps2*1000);

	


	
	
	checkResult(Ans_cpu_sorted_data, Ans_gpu_sorted_data, N);

	free(data); 
	free(cpu_sorted_data); 
	free(gpu_sorted_data); 
	free(Ans_cpu_sorted_data);
	free(Ans_gpu_sorted_data);

	return 0;
	
} 

