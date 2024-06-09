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
	MPI_Request req;
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
		//ע���ܲ�����ȥ�ɾ�
		if(k>=average_row*rank && k<average_row*(rank+1))
		{
            for(int j=k+1;j<n;j++)
                A[k][j] = A[k][j]/A[k][k];
            A[k][k] = 1.0;
		}
		MPI_Ibcast(&A[k][0],n,MPI_FLOAT,n/average_row,MPI_COMM_WORLD,&req);
		MPI_Wait(&req,&status);
		//ע��˴�
        for(int i=max(average_row*rank,k+1);i<average_row*(rank+1);i++)
        {
            for(int j=k+1;j<n;j++)
            A[i][j] = A[i][j] - A[i][k]*A[k][j];
        A[i][k] = 0;
        }
    }
	//����һ������
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
