#include <iostream>
#include <nmmintrin.h>
#include <windows.h>
#include <stdlib.h>
#include "mpi.h"
#include <omp.h>
#include <tchar.h>
#include <stdio.h>

using namespace std;

const int N = 4000;

float** m;

void m_reset(int);
void m_gauss(int);
void m_gauss_simd(int);
void m_gauss_mpi_1(int, float**, int, int);
void m_gauss_mpi_2(int, float**, int, int);
void m_gauss_mpi_3(int, float**, int, int);
void m_gauss_mpi_4(int, float**, int, int);
void m_gauss_mpi_6(int, float**, int, int);

int PROCESS_THREADS = 8;

int main()
{
    for (int i = 0; i < N; i++)
        m = new float* [N];
    for (int i = 0; i < N; i++)
        m[i] = new float[N];
    long long head, tail, freq; // timers
    int step = 10;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    for (int n = 10; n <= 1000; n += step)
    {
    int n = 1000;
        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        m_gauss(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "n: " << n << " 串行算法用时" << (tail - head) * 1000.0 / freq << "ms" << endl;
        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        m_gauss_simd(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << (tail - head) * 1000.0 / freq << "ms" << endl;
        if (n == 100) step = 100;
    }
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    step = 10;
    for (int n = 10; n <= 1000; n += step)
    {
        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        m_gauss_mpi_1(n, m, rank, size);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "n: " << n << " 普通MPI用时" << (tail - head) * 1000.0 / freq << "ms" << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        m_gauss_mpi_6(n, m, rank, size);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "n: " << n << " 负载均衡MPI用时" << (tail - head) * 1000.0 / freq << "ms" << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        m_gauss_mpi_2(n, m, rank, size);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "n: " << n << " 循环划分MPI用时" << (tail - head) * 1000.0 / freq << "ms" << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        m_gauss_mpi_3(n, m, rank, size);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "n: " << n << " SIMD+MPI用时" << (tail - head) * 1000.0 / freq << "ms" << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        m_gauss_mpi_4(n, m, rank, size);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "n: " << n << " OpenMP+MPI用时" << (tail - head) * 1000.0 / freq << "ms" << endl;

        if (n == 100) step = 100;
    }
    MPI_Finalize();
    return 0;
}

//初始化矩阵元素
void m_reset(int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
            m[i][j] = 0;
        m[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            m[i][j] = rand();
    }
    for (int k = 0; k < n; k++)
        for (int i = k + 1; i < n; i++)
            for (int j = 0; j < n; j++)
                m[i][j] += m[k][j];
}

//串行普通高斯消去算法
void m_gauss(int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;

        }
    }
}

//乘法和除法部分全部向量化、不对齐
void m_gauss_simd(int n)
{
    __m128 vt, va, vaik, vakj, vaij, vx;
    for (int k = 0; k < n; k++) {
        vt = _mm_set_ps1(m[k][k]);
        int j;
        for (j = k + 1; j + 4 <= n; j += 4) {
            va = _mm_loadu_ps(&m[k][j]);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&m[k][j], va);
        }
        if (j < n) {
            for (; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            vaik = _mm_set_ps1(m[i][k]);
            int j;
            for (j = k + 1; j + 4 <= n; j += 4) {
                vakj = _mm_loadu_ps(&m[k][j]);
                vaij = _mm_loadu_ps(&m[i][j]);
                vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&m[i][j], vaij);
            }
            if (j < n) {
                for (; j < n; j++) {
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                }
            }
            m[i][k] = 0;
        }
    }
}

void m_gauss_mpi_1(int n, float **m, int rank, int size)
{
    if (rank != size - 1)
    {
        for (int k = 0; k < n; k++)
        {
            if (k <= rank * (n - n % size) / size + (n - n % size) / size - 1 && k >= rank * (n - n % size) / size)
            {
                for (int j = k + 1; j < n; j++)
                {
                    m[k][j] = m[k][j] / m[k][k];
                }
                m[k][k] = 1.0;
                for (int j = 0; j < size; j++)
                    if (j != rank)
                        MPI_Send(&m[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Recv(&m[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = rank * (n - n % size) / size; i < rank * (n - n % size) / size + (n - n % size) / size; i++)
            {
                for (int j = k + 1; j < n; j++)
                {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                m[i][k] = 0;
            }
        }
    }
    else
    {
        for (int k = 0; k < n; k++)
        {
            if (k <= n - 1 && k >= rank * (n - n % size) / size)
            {
                for (int j = k + 1; j < n; j++)
                {
                    m[k][j] = m[k][j] / m[k][k];
                }
                m[k][k] = 1.0;
                for (int j = 0; j < size; j++)
                    if (j != rank)
                        MPI_Send(&m[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Recv(&m[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = rank * (n - n % size) / size; i < n; i++)
            {
                for (int j = k + 1; j < n; j++)
                {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                m[i][k] = 0;
            }
        }
    }
}

void m_gauss_mpi_2(int n, float** m, int rank, int size)
{
    if (rank != size - 1)
    {
        for (int k = 0; k < n; k++)
        {
            if (k <= rank * (n - n % size) / size + (n - n % size) / size - 1 && k >= rank * (n - n % size) / size)
            {
                for (int j = 0; j < size; j++)
                    if (j != rank)
                        MPI_Send(&m[k][k], 1, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Recv(&m[k][k], 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = rank * (n - n % size) / size; j < rank * (n - n % size) / size + (n - n % size) / size; j++)
            {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0;
            for (int i = k + 1; i < n; i++)
            {
                for (int j = rank * (n - n % size) / size; j < rank * (n - n % size) / size + (n - n % size) / size; j++)
                {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                m[i][k] = 0;
            }
        }
    }
    else
    {
        for (int k = 0; k < n; k++)
        {
            if (k <= n - 1 && k >= rank * (n - n % size) / size)
            {
                for (int j = 0; j < size; j++)
                    if (j != rank)
                        MPI_Send(&m[k][k], 1, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Recv(&m[k][k], 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = rank * (n - n % size) / size; j < rank * (n - n % size) / size + (n - n % size) / size; j++)
            {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0;
            for (int i = k + 1; i < n; i++)
            {
                for (int j = rank * (n - n % size) / size; j < n; j++)
                {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                m[i][k] = 0;
            }
        }
    }
}

void m_gauss_mpi_3(int n, float** m, int rank, int size)
{
    __m128 vt, va, vaik, vakj, vaij, vx;
    if (rank != size - 1)
    {
        for (int k = 0; k < n; k++)
        {
            if (k <= rank * (n - n % size) / size + (n - n % size) / size - 1 && k >= rank * (n - n % size) / size)
            {
                vt = _mm_set_ps1(m[k][k]);
                int j;
                for (j = k + 1; j + 4 <= n; j += 4) {
                    va = _mm_loadu_ps(&m[k][j]);
                    va = _mm_div_ps(va, vt);
                    _mm_storeu_ps(&m[k][j], va);
                }
                if (j < n) {
                    for (; j < n; j++) {
                        m[k][j] = m[k][j] / m[k][k];
                    }
                }
                m[k][k] = 1.0;
                for (int j = 0; j < size; j++)
                    if (j != rank)
                        MPI_Send(&m[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Recv(&m[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = rank * (n - n % size) / size; i < rank * (n - n % size) / size + (n - n % size) / size; i++)
            {
                vaik = _mm_set_ps1(m[i][k]);
                int j;
                for (j = k + 1; j + 4 <= n; j += 4) {
                    vakj = _mm_loadu_ps(&m[k][j]);
                    vaij = _mm_loadu_ps(&m[i][j]);
                    vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);
                    _mm_storeu_ps(&m[i][j], vaij);
                }
                if (j < n) {
                    for (; j < n; j++) {
                        m[i][j] = m[i][j] - m[k][j] * m[i][k];
                    }
                }
                m[i][k] = 0;
            }
        }
    }
    else
    {
        for (int k = 0; k < n; k++)
        {
            if (k <= n - 1 && k >= rank * (n - n % size) / size)
            {
                vt = _mm_set_ps1(m[k][k]);
                int j;
                for (j = k + 1; j + 4 <= n; j += 4) {
                    va = _mm_loadu_ps(&m[k][j]);
                    va = _mm_div_ps(va, vt);
                    _mm_storeu_ps(&m[k][j], va);
                }
                if (j < n) {
                    for (; j < n; j++) {
                        m[k][j] = m[k][j] / m[k][k];
                    }
                }
                m[k][k] = 1.0;
                for (int j = 0; j < size; j++)
                    if (j != rank)
                        MPI_Send(&m[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Recv(&m[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = rank * (n - n % size) / size; i < n; i++)
            {
                vaik = _mm_set_ps1(m[i][k]);
                int j;
                for (j = k + 1; j + 4 <= n; j += 4) {
                    vakj = _mm_loadu_ps(&m[k][j]);
                    vaij = _mm_loadu_ps(&m[i][j]);
                    vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);
                    _mm_storeu_ps(&m[i][j], vaij);
                }
                if (j < n) {
                    for (; j < n; j++) {
                        m[i][j] = m[i][j] - m[k][j] * m[i][k];
                    }
                }
                m[i][k] = 0;
            }
        }
    }
}

void m_gauss_mpi_4(int n, float** m, int rank, int size)
{
    if (rank != size - 1)
    {
        #pragma omp parallel for
        for (int k = 0; k < n; k++)
        {
            if (k <= rank * (n - n % size) / size + (n - n % size) / size - 1 && k >= rank * (n - n % size) / size)
            {
                #pragma omp parallel for
                for (int j = k + 1; j < n; j++)
                {
                    m[k][j] = m[k][j] / m[k][k];
                }
                m[k][k] = 1.0;
                for (int j = 0; j < size; j++)
                    if (j != rank)
                        MPI_Send(&m[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Recv(&m[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            #pragma omp parallel for
            for (int i = rank * (n - n % size) / size; i < rank * (n - n % size) / size + (n - n % size) / size; i++)
            {
                #pragma omp parallel for
                for (int j = k + 1; j < n; j++)
                {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                m[i][k] = 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int k = 0; k < n; k++)
        {
            if (k <= n - 1 && k >= rank * (n - n % size) / size)
            {
                #pragma omp parallel for
                for (int j = k + 1; j < n; j++)
                {
                    m[k][j] = m[k][j] / m[k][k];
                }
                m[k][k] = 1.0;
                for (int j = 0; j < size; j++)
                    if (j != rank)
                        MPI_Send(&m[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Recv(&m[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            #pragma omp parallel for
            for (int i = rank * (n - n % size) / size; i < n; i++)
            {
                #pragma omp parallel for
                for (int j = k + 1; j < n; j++)
                {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                m[i][k] = 0;
            }
        }
    }
}

void m_gauss_mpi_6(int n, float** m, int rank, int size)
{
    if (rank < n / size && rank != size - 1)
    {
        for (int k = 0; k < n; k++)
        {
            if (k <= rank * (n - n % size) / size + (n - n % size) / size && k >= rank * (n - n % size) / size)
            {
                for (int j = k + 1; j < n; j++)
                {
                    m[k][j] = m[k][j] / m[k][k];
                }
                m[k][k] = 1.0;
                for (int j = 0; j < size; j++)
                    if (j != rank)
                        MPI_Send(&m[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Recv(&m[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = rank * (n - n % size) / size; i <= rank * (n - n % size) / size + (n - n % size) / size; i++)
            {
                for (int j = k + 1; j < n; j++)
                {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                m[i][k] = 0;
            }
        }
    }
    else
    {
        if (rank == size - 1)
        {
            for (int k = 0; k < n; k++)
            {
                if (k <= n - 1 && k >= rank * (n - n % size) / size)
                {
                    for (int j = k + 1; j < n; j++)
                    {
                        m[k][j] = m[k][j] / m[k][k];
                    }
                    m[k][k] = 1.0;
                    for (int j = 0; j < size; j++)
                        if (j != rank)
                            MPI_Send(&m[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
                }
                else
                    MPI_Recv(&m[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = rank * (n - n % size) / size; i < n; i++)
                {
                    for (int j = k + 1; j < n; j++)
                    {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                    m[i][k] = 0;
                }
            }
        }
        else
        {
            for (int k = 0; k < n; k++)
            {
                if (k <= rank * (n - n % size) / size + (n - n % size) / size - 1 && k >= rank * (n - n % size) / size)
                {
                    for (int j = k + 1; j < n; j++)
                    {
                        m[k][j] = m[k][j] / m[k][k];
                    }
                    m[k][k] = 1.0;
                    for (int j = 0; j < size; j++)
                        if (j != rank)
                            MPI_Send(&m[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
                }
                else
                    MPI_Recv(&m[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = rank * (n - n % size) / size; i < rank * (n - n % size) / size + (n - n % size) / size; i++)
                {
                    for (int j = k + 1; j < n; j++)
                    {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                    m[i][k] = 0;
                }
            }
        }
    }
}