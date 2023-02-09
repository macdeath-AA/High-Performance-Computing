#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void grid(int nx, int nxglob, int istglob, int ienglob, double xstglob, double xenglob, double *x, double *dx)
{
    int i, iglob;

    *dx = (xenglob - xstglob) / (double)(nxglob - 1); 

    for (i = 0; i < nx; i++)
    {
        iglob = istglob + i;
        x[i] = xstglob + (double)iglob * (*dx);
    }
}

void enforce_bcs(int nx, int nxglob, int istglob, int ienglob, double *x, double *T)
{
    if (istglob == 0)
        T[0] = -1.0;

    if (ienglob == nxglob - 1)
        T[nx - 1] = 1.0;
}

void get_exact_solution(int nx, double time, double *x, double *Texact)
{
    int i;

    for (i = 0; i < nx; i++)
        Texact[i] = erf((x[i] - 0.5) / 2.0 / sqrt(time));
}

void set_initial_condition(int nx, int nxglob, int istglob, int ienglob, double time, double *x, double *T)
{
    int i;

    get_exact_solution(nx, time, x, T);

    enforce_bcs(nx, nxglob, istglob, ienglob, x, T); 
}

void get_rhs(int nx, int nxglob, int istglob, int ienglob, double dx, double xleftghost, double xrghtghost, double *x, double *T, double *rhs)
{
    int i;
    double dxsq = dx * dx;

    
    for (i = 1; i < nx - 1; i++)
        rhs[i] = (T[i + 1] + T[i - 1] - 2.0 * T[i]) / dxsq;

    i = 0;
    if (istglob == 0) 
        rhs[i] = 0.0;
    else
        rhs[i] = (T[i + 1] + xleftghost - 2.0 * T[i]) / dxsq; 

    i = nx - 1;
    if (ienglob == nxglob - 1) 
        rhs[i] = 0.0;
    else
        rhs[i] = (xrghtghost + T[i - 1] - 2.0 * T[i]) / dxsq; 
}

void halo_exchange_1d_x(int rank, int size, int nx, double *x, double *T, double *xleftghost, double *xrghtghost)
{
    MPI_Status status;
    FILE *fid;
    char debugfname[100];
    int left_nb, rght_nb;

    if (rank > 0)
        left_nb = rank - 1;
    else
        left_nb = MPI_PROC_NULL;

    if (rank < size - 1)
        rght_nb = rank + 1;
    else
        rght_nb = MPI_PROC_NULL;

    MPI_Recv(xrghtghost, 1, MPI_DOUBLE, rght_nb, 0, MPI_COMM_WORLD, &status);
    MPI_Send(&T[0], 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD);

    MPI_Recv(xleftghost, 1, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD, &status);
    MPI_Send(&T[nx - 1], 1, MPI_DOUBLE, rght_nb, 0, MPI_COMM_WORLD);

    
}

double timestep_Euler(int rank, int size, int nx, int nxglob, int istglob, int ienglob, double dt, double dx, double *x, double *T, double *rhs)
{

    int i;
    double xleftghost, xrghtghost;
    double t_comm;

    t_comm = MPI_Wtime();
    xleftghost = 0.0;
    xrghtghost = 0.0;
    halo_exchange_1d_x(rank, size, nx, x, T, &xleftghost, &xrghtghost);
    t_comm = MPI_Wtime() - t_comm;

    get_rhs(nx, nxglob, istglob, ienglob, dx, xleftghost, xrghtghost, x, T, rhs);

    for (i = 0; i < nx; i++)
        T[i] = T[i] + dt * rhs[i];

    enforce_bcs(nx, nxglob, istglob, ienglob, x, T);
    return t_comm;
}

void output_soln(int rank, int nx, int it, double tcurr, double *x, double *T, double *Tex)
{
    int i;
    FILE *fp;
    char fname[100];
    sprintf(fname,"/clhome/me20btech11001/hpc/assn2/temp_10/T_x_%04d_%04d.txt",it,rank);
   // sprintf(fname, "T_x_%04d_%04d.dat", it, rank);

    fp = fopen(fname, "w");
    for (i = 0; i < nx; i++)
        fprintf(fp, "%.15e %.15e %.15e\n", x[i], T[i], Tex[i]);
    fclose(fp);
}

void output_error_norm(int rank, int size, int nx, int nxglob, double time, double *T, double *Tex)
{
    int i;
    FILE *fp;
    char fname[100];
    double l2err, l2err_loc;

    l2err_loc = 0.0;

    for (i = 0; i < nx; i++)
    {
        l2err_loc += (T[i] - Tex[i]) * (T[i] - Tex[i]);
    }
    MPI_Reduce(&l2err_loc, &l2err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        l2err = sqrt(l2err);
       // sprintf(fname, "L2_norm_%d.dat", size);
        sprintf(fname,"/clhome/me20btech11001/hpc/assn2/temp_10/L2_norm_%d.txt",size);

        fp = fopen(fname, "w");
        fprintf(fp, "%.15e\n", l2err);
        fclose(fp);
    }
}

int main(int argc, char **argv)
{

    int nx;
    double *x, *T, *rhs, *Texact;
    double tst, ten, xst, xen, dx, dt, tcurr, xlen, t_print;
    int i, it, num_time_steps, it_print;
    char debugfname[100], filename[100];
    FILE *fid, *fp, *file_t;


    int rank, size;
    int nxglob, istglob, ienglob;
    double xstglob, xenglob;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        fp = fopen("input.in", "r");
        fscanf(fp, "%d\n", &nxglob);
        fscanf(fp, "%lf %lf\n", &xstglob, &xenglob);
        fscanf(fp, "%lf %lf\n", &tst, &ten);
        fscanf(fp, "%lf %lf\n", &dt, &t_print);
        fclose(fp);

        

        nx = nxglob / size;                        
        xlen = (xenglob - xstglob) / (double)size; 

        num_time_steps = (int)((ten - tst) / dt) + 1; 
        it_print = (int)(t_print / dt);               
    }

    int *sendarr_int;
    sendarr_int = malloc(4 * sizeof(int));
    if (rank == 0)
    {
        sendarr_int[0] = nxglob;
        sendarr_int[1] = nx;
        sendarr_int[2] = num_time_steps;
        sendarr_int[3] = it_print;
    }
    MPI_Bcast(sendarr_int, 4, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        nxglob = sendarr_int[0];
        nx = sendarr_int[1];
        num_time_steps = sendarr_int[2];
        it_print = sendarr_int[3];
    }
    free(sendarr_int);

    double *sendarr_dbl;
    sendarr_dbl = malloc(7 * sizeof(double));
    if (rank == 0)
    {
        sendarr_dbl[0] = tst;
        sendarr_dbl[1] = ten;
        sendarr_dbl[2] = dt;
        sendarr_dbl[3] = t_print;
        sendarr_dbl[4] = xlen;
        sendarr_dbl[5] = xstglob;
        sendarr_dbl[6] = xenglob;
    }
    MPI_Bcast(sendarr_dbl, 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        tst = sendarr_dbl[0];
        ten = sendarr_dbl[1];
        dt = sendarr_dbl[2];
        t_print = sendarr_dbl[4];
        xlen = sendarr_dbl[4];
        xstglob = sendarr_dbl[5];
        xenglob = sendarr_dbl[6];
    }
    free(sendarr_dbl);

    istglob = rank * (nxglob / size); 
    ienglob = (rank + 1) * (nxglob / size) - 1; 


    xst = xstglob + rank * xlen; 
    xen = xst + xlen;            

    x = (double *)malloc(nx * sizeof(double)); 
    T = (double *)malloc(nx * sizeof(double));
    rhs = (double *)malloc(nx * sizeof(double));
    Texact = (double *)malloc(nx * sizeof(double));

    grid(nx, nxglob, istglob, ienglob, xstglob, xenglob, x, &dx); 

    set_initial_condition(nx, nxglob, istglob, ienglob, tst, x, T); 

    get_exact_solution(nx, tst, x, Texact); 
    output_soln(rank, nx, 0, tst, x, T, Texact);

    

    double *t_comm = (double *)malloc(num_time_steps * sizeof(double));
    double *t_comp = (double *)malloc(num_time_steps * sizeof(double));
    double *t_tot = (double *)malloc(num_time_steps * sizeof(double));

    for (it = 0; it < num_time_steps; it++)
    {
        tcurr = tst + (double)it * dt;
        if (rank == 0)
            printf("Working on time step no. %d, time = %lf\n", it, tcurr);

        t_tot[it] = MPI_Wtime();
        t_comm[it] = timestep_Euler(rank, size, nx, nxglob, istglob, ienglob, dt, dx, x, T, rhs);
        t_tot[it] = MPI_Wtime() - t_tot[it];
        t_comp[it] = t_tot[it] - t_comm[it];

        if ((it % it_print == 0) && (it != 0))
        {
            if (rank == 0)
                printf("Writing solution at time step no. %d, time = %lf\n", it, tcurr);
            get_exact_solution(nx, tcurr, x, Texact);
            output_soln(rank, nx, it + 1, tcurr, x, T, Texact);
        }
    }

    
    output_error_norm(rank, size, nx, nxglob, tcurr, T, Texact);

  
    sprintf(filename,"/clhome/me20btech11001/hpc/assn2/temp_10/times.txt");
    printf("%s\n",filename);
    file_t = fopen(filename,"w");

    int j;
    for(j = 0; j < num_time_steps; j++)
    {
       fprintf(file_t,"%.15e %.15e %.15e\n", t_tot[j], t_comm[j], t_comp[j]);
    }
    fclose(file_t);


    free(rhs);
    free(T);
    free(x);
    free(Texact);

    MPI_Finalize();
    return 0;
}
