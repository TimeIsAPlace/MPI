#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <mpi.h>
#include <cmath>
#include <semaphore.h>
#include <pthread.h>
using namespace std;
const int n=1024;
const int mpi_num = 1;
int average_row;
float A[n][n];
const int NUM_THREADS=7;
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];
sem_t sem_workerend[NUM_THREADS];
typedef struct
{
    int t_id;
	int rank;
}threadParam_t;
void *threadFunc(void *param)
{
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p->t_id;
	int rank = p->rank;
	for(int k=0;k<n;k++)
	{
		int total_begin = max(average_row*rank,k+1);
	    int total_end = average_row*(rank+1);
		int avgnum = (total_end-total_begin)/NUM_THREADS;
    	int begin = total_begin + avgnum*t_id;
	    int end = total_begin + avgnum*(t_id+1);
	    if(t_id==6)
		    end = total_end;
		sem_wait(&sem_workerstart[t_id]);
		if(avgnum>0)
		{
	    for(int i=begin;i<end;i++)
	    {
            for(int j=k+1;j<n;j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
	    }
		}
		sem_post(&sem_main);
		sem_wait(&sem_workerend[t_id]);

	}
    pthread_exit(NULL);
}
int main(int argc,char* argv[])
{
	int buf[2048],provided;
    struct timespec startTime;
    struct timespec endTime;
	time_t dsec;
    long dnsec;
  
	int rank,size;
        MPI_Status status;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
	if(provided<MPI_THREAD_FUNNELED)
		MPI_Abort(MPI_COMM_WORLD,1);
	timespec_get(&startTime,TIME_UTC);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	average_row = n/size;
    double tb = MPI_Wtime();

	sem_init(&sem_main,0,0);
	for(int i=0;i<NUM_THREADS;i++)
	{
		sem_init(&sem_workerstart[i],0,0);
	    sem_init(&sem_workerend[i],0,0);
	}
	pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
	for(int t_id=0;t_id<NUM_THREADS;t_id++)
        {
            param[t_id].t_id = t_id;
			param[t_id].rank = rank;
        }
    for(int t_id=0;t_id<NUM_THREADS;t_id++)
        pthread_create(&handles[t_id],NULL,threadFunc,(void*)(&param[t_id]));

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
            for(int j=k+1;j<n;j++)
                A[k][j] = A[k][j]/A[k][k];
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
        for(int i=0;i<NUM_THREADS;i++)
			sem_post(&sem_workerstart[i]);
		for(int i=0;i<NUM_THREADS;i++)
			sem_wait(&sem_main);
		for(int i=0;i<NUM_THREADS;i++)
			sem_post(&sem_workerend[i]);
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
