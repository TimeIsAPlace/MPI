#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <mpi.h>
#include <cmath>
using namespace std;
const int N = 1000;
int step = 10;
const int mpi_num = 1;
float A[N][N];
void m_reset(int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < i; j++)
			A[i][j] = 0;
		A[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			A[i][j] = rand();
	}
	for (int k = 0; k < n; k++)
		for (int i = k + 1; i < n; i++)
			for (int j = 0; j < n; j++)
				A[i][j] += A[k][j];
}
int main(int argc, char* argv[])
{
	struct timespec startTime;
	struct timespec endTime;
	time_t dsec;
	long dnsec;
	int rank, size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	timespec_get(&startTime, TIME_UTC);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	for (int n = 10; n <= 1000; n += step)
	{
		m_reset(n);
		int average_row = n / size;
		double tb = MPI_Wtime();
		for (int k = 0; k < n; k++)
		{
			//注意能不能消去干净
			if (k >= average_row * rank && k < average_row * (rank + 1))
			{
				for (int j = k + 1; j < n; j++)
					A[k][j] = A[k][j] / A[k][k];
				A[k][k] = 1.0;
				for (int j = 0; j < size; j++)
				{
					if (rank != j)
						MPI_Send(&A[k][0], n, MPI_FLOAT, j, j, MPI_COMM_WORLD);
				}
			}
			else
			{
				MPI_Recv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, rank, MPI_COMM_WORLD, &status);
			}
			for (int i = max(average_row * rank, k + 1); i < average_row * (rank + 1); i++)
			{
				for (int j = k + 1; j < n; j++)
					A[i][j] = A[i][j] - A[i][k] * A[k][j];
				A[i][k] = 0;
			}
		}
		if (rank == 0)
		{
			for (int i = 1; i < size; i++)
			{
				MPI_Recv(&A[average_row * i][0], n * average_row, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
			}
		}
		else
		{
			MPI_Send(&A[average_row * rank][0], n * average_row, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
		}
		timespec_get(&endTime, TIME_UTC);
		dsec = endTime.tv_sec - startTime.tv_sec;
		dnsec = endTime.tv_nsec - startTime.tv_nsec;
		double te = MPI_Wtime();
		if (rank == 0)
		{
			cout << "n: " << n << "用时";
			printf("%llu.%09llus\n", dsec, dnsec);
		}
		cout << "n: " << n << "用时";
		printf("Time:%fs\n", te - tb);
		if (n == 100)
			step = 100;
	}
	MPI_Finalize();
	return 0;
}
