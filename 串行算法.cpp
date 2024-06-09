#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
using namespace std;
const int n=1000;
int main()
{
    float A[n][n];
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<i;j++)
            A[i][j] = 0;
        A[i][i] = 1;
        for(int j=i+1;j<n;j++)
            A[i][j] = double(rand()%10);
    }
    for(int k=0;k<n;k++)
        for(int i=k+1;i<n;i++)
        for(int j=0;j<n;j++)
        A[i][j] += A[k][j];

    struct timespec startTime;
    struct timespec endTime;
    timespec_get(&startTime,TIME_UTC);
    for(int k=0;k<n;k++)
    {
        for(int j=k+1;j<n;j++)
            A[k][j] = A[k][j]/A[k][k];
        A[k][k] = 1.0;
        for(int i=k+1;i<n;i++)
        {
            for(int j=k+1;j<n;j++)
            A[i][j] = A[i][j] - A[i][k]*A[k][j];
        A[i][k] = 0;
        }
    }
    timespec_get(&endTime,TIME_UTC);
    time_t dsec=endTime.tv_sec - startTime.tv_sec;
    long dnsec=endTime.tv_nsec-startTime.tv_nsec;
    printf("%llu.%09llus\n",dsec,dnsec);
    cout<<endl;

    return 0;
}
