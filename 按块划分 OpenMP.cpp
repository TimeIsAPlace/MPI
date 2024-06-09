#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
using namespace std;
int max(int a,int b)
{
	if(a>b)
		return a;
	else
		return b;
}
const int n=1024;
const int mpi_num = 1;
const int NUM_THREADS = 4;
int main(int argc,char* argv[])
{
    float A[n][n];

    struct timespec startTime;
    struct timespec endTime;
	time_t dsec;
    long dnsec;
  
	int buf[2000],provided;
	int rank,size;
        MPI_Status status;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
	if(provided<MPI_THREAD_FUNNELED)
		MPI_Abort(MPI_COMM_WORLD,1);
	timespec_get(&startTime,TIME_UTC);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	int average_row = n/size;
        double tb = MPI_Wtime();
	if(rank==0)
	{
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
				A[i][j] = 0;
			A[i][i] = 1;
		}
	}
	if(rank==0)
	{
		for(int i=1;i<size;i++)
		    MPI_Send(&A[0][0],n*n,MPI_FLOAT,i,i,MPI_COMM_WORLD);
	}
	else
	{
		MPI_Recv(&A[0][0],n*n,MPI_FLOAT,0,rank,MPI_COMM_WORLD,&status);
	}
	double tmp;
	int i,j,k;
    #pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp,average_row,rank)
	{
		MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        int average_row = n/size;
       	for(int k=0;k<n;k++)
    {
		//注意能不能消去干净
		if(k>=average_row*rank && k<average_row*(rank+1))
		{
			#pragma omp single
			{
				tmp = A[k][k];
            for(int j=k+1;j<n;j++)
                A[k][j] = A[k][j]/tmp;
            A[k][k] = 1.0;
			}
			for(int j=0;j<size;j++)
			{
                #pragma omp single
				if(rank != j)
				{
				    MPI_Send(&A[k][0],n,MPI_FLOAT,j,j,MPI_COMM_WORLD);
				}
			}
		}
		else
		{
             #pragma omp single
			{	
			MPI_Recv(&A[k][0],n,MPI_FLOAT,k/average_row,rank,MPI_COMM_WORLD,&status);
			}
		}
		//注意此处
	
        for(int i=max(average_row*rank,k+1);i<average_row*(rank+1);i++)
        {
			tmp = A[i][k];
            #pragma omp for
            for(int j=k+1;j<n;j++)
	    {
                A[i][j] = A[i][j] - tmp*A[k][j];
	    }
        A[i][k] = 0;
        }
    }
	}
	//还差一个汇总
	if(rank==0)
	{
		for(int i=1;i<size;i++)
		{
			MPI_Recv(&A[average_row*i][0],n*average_row,MPI_FLOAT,i,i,MPI_COMM_WORLD,&status);
		}
	}
	else
	{
		MPI_Send(&A[average_row*rank][0],n*average_row,MPI_FLOAT,0,rank,MPI_COMM_WORLD);
	}
	timespec_get(&endTime,TIME_UTC);
    dsec=endTime.tv_sec - startTime.tv_sec;
    dnsec=endTime.tv_nsec-startTime.tv_nsec;
    double te = MPI_Wtime();
	if(rank==0)
	{
    printf("%llu.%09llus\n",dsec,dnsec);
    
	}
printf("Time:%f s\n",te-tb);
	MPI_Finalize();
    return 0;
}
