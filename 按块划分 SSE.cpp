#include <pmmintrin.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <mpi.h>
#include <cmath>
using namespace std;
const int n=1024;
const int mpi_num = 1;
int main(int argc,char* argv[])
{
    float A[n][n];
	__m128 vt;
	__m128 va;
	__m128 vaik;
	__m128 vakj;
    __m128 vaij;
    __m128 vx;

    struct timespec startTime;
    struct timespec endTime;
	time_t dsec;
    long dnsec;
  
	int rank,size;
        MPI_Status status;
	MPI_Init(&argc,&argv);
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
    for(int k=0;k<n;k++)
    {
		//注意能不能消去干净
		if(k>=average_row*rank && k<average_row*(rank+1))
		{
            vt=_mm_set1_ps(A[k][k]);
            for(int j=k+1;j+4<n;j+=4)
            {
                va=_mm_loadu_ps(&A[k][j]);
                va=_mm_div_ps(va,vt);
                _mm_storeu_ps(&A[k][j],va);
            }
			A[k][k] = 1.0;
			for(int j=0;j<size;j++)
			{
				if(rank != j)
				    MPI_Send(&A[k][0],n,MPI_FLOAT,j,j,MPI_COMM_WORLD);
			}
		}
		else
		{
			MPI_Recv(&A[k][0],n,MPI_FLOAT,k/average_row,rank,MPI_COMM_WORLD,&status);
		}
		//注意此处
        for(int i=max(average_row*rank,k+1);i<average_row*(rank+1);i++)
        {
            vaik=_mm_set1_ps(A[i][k]);
            for(int j=k+1;j+4<n;j+=4)
            {
                vakj=_mm_loadu_ps(&A[k][j]);
                vaij=_mm_loadu_ps(&A[i][j]);
                vx=_mm_mul_ps(vakj,vaik);
                vaij=_mm_sub_ps(vaij,vx);
                _mm_storeu_ps(&A[i][j],vaij);
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
    cout<<endl;
	}
printf("Time:%f s\n",te-tb);
	MPI_Finalize();
    return 0;
}
