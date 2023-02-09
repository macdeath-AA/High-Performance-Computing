#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

void mat_diff(int **matrix1, int **matrix2, int n)
{
    int i, j;
    double error[n];
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            error[i] = abs(matrix1[i][j] - matrix2[i][j]);
        }
    }
    double max_val = error[0];
    for (i = 0; i < n; i++)
    {
        if (error[i] > max_val)
        {
            max_val = error[i];
        }
    }
    printf("Error in matrix: %.4f\n", max_val);
}

int main()
{
    int n = 800, p = 1, size;
    int **a, **b, **c_ser, **c_par;
    int i, j, k;

    size = n * n;

    omp_set_num_threads(p);

    a = (int **)malloc(n * sizeof(int *));
    b = (int **)malloc(n * sizeof(int *));
    c_par = (int **)malloc(n * sizeof(int *));
    c_ser = (int **)malloc(n * sizeof(int *));

    for (i = 0; i < n; i++)
    {
        a[i] = (int *)malloc(n * sizeof(int));
        b[i] = (int *)malloc(n * sizeof(int));
        c_par[i] = (int *)malloc(n * sizeof(int));
        c_ser[i] = (int *)malloc(n * sizeof(int));
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a[i][j] = -1;
            b[i][j] = 1;
            c_par[i][j] = 0;
            c_ser[i][j] = 0;
        }
    }

    double ser_start = omp_get_wtime();
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            c_ser[i][j] = 0;
            for (k = 0; k < n; k++)
            {
                c_ser[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    double ser_end = omp_get_wtime();
    double ser_elapsed = (ser_end - ser_start);
    printf("Time reports for %d threads.\n", p);
    printf("Serial time: %.6f seconds.\n", ser_elapsed);

    double par_start = omp_get_wtime();

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c_par[i][j] = 0;
            int sum = 0;
#pragma omp parallel
            {
#pragma omp for reduction(+ \
                          : sum)
                for (int k = 0; k < n; k++)
                {
                    sum += a[i][k] * b[k][j];
                }
                c_par[i][j] = sum;
            }
        }
    }

    double par_end = omp_get_wtime();
    double par_elapsed = (par_end - par_start);
    printf("Parallel time: %.6f seconds.\n", par_elapsed);

    mat_diff(c_par, c_ser, 800);

    for (int i = 0; i < n; i++)
    {
        free(a[i]);
        free(b[i]);
        free(c_par[i]);
        free(c_ser[i]);
    }
    free(a);
    free(b);
    free(c_par);
    free(c_ser);

    return 0;
}
