/* ================================================================
 * 1d_init_euler.c
 *
 * Gradient computation w.r.t. initial conditions (x0, v0)
 * for a 1-D damped oscillator via the forward Euler adjoint method.
 *
 * ODE:  u' = f(u),  u = (x, v)
 *   f[0] =  v
 *   f[1] = -2*zeta*omega*v - omega^2*x
 *
 * Cost:  C = (1/2) * (x_N - x_obs_N)^2   (terminal cost only)
 *
 * Adjoint scheme: implicit Euler (symplectic pair of forward Euler)
 *   (I + h*A^T) * lambda_{k-1} = lambda_k
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ----------------------------------------------------------------
 * Problem parameters
 * ---------------------------------------------------------------- */
#define ZETA   0.1   /* damping ratio */
#define OMEGA  2.0   /* natural frequency (true value for observation) */
#define X0     1.0   /* initial displacement */
#define V0     0.0   /* initial velocity */
#define T_END  10.0  /* simulation end time */

/* ================================================================
 * Analytical solution
 * ================================================================ */

/* x(t) starting from (X0, V0) */
static double x_exact(double t, double omega)
{
    double sq = sqrt(1.0 - ZETA*ZETA);
    double wd = omega * sq;
    return exp(-ZETA*omega*t)
           * (X0*cos(wd*t) + (V0 + ZETA*omega*X0)/wd * sin(wd*t));
}

/* x(t) starting from arbitrary (x0, v0) */
static double x_exact_with_init(double t, double x0, double v0, double omega)
{
    double sq = sqrt(1.0 - ZETA*ZETA);
    double wd = omega * sq;
    double e  = exp(-ZETA*omega*t);
    return e * (x0*cos(wd*t) + (v0 + ZETA*omega*x0)/wd * sin(wd*t));
}

/* d x(t) / d x0  (analytical) */
static double dx_dx0_analytical(double t, double omega)
{
    double sq = sqrt(1.0 - ZETA*ZETA);
    double wd = omega * sq;
    double e  = exp(-ZETA*omega*t);
    return e * (cos(wd*t) + (ZETA*omega)/wd * sin(wd*t));
}

/* d x(t) / d v0  (analytical) */
static double dx_dv0_analytical(double t, double omega)
{
    double sq = sqrt(1.0 - ZETA*ZETA);
    double wd = omega * sq;
    double e  = exp(-ZETA*omega*t);
    return e * (1.0/wd * sin(wd*t));
}

/* dC/dx0  (analytical reference) */
static double grad_x0_analytical(double omega, double h, int N,
                                  const double *x_obs)
{
    double T  = N * h;
    double xT = x_exact(T, omega);
    return (xT - x_obs[N]) * dx_dx0_analytical(T, omega);
}

/* ================================================================
 * ODE right-hand side and Jacobian
 * ================================================================ */

/* f(u) = (v, -omega^2*x - 2*zeta*omega*v) */
static void f_rhs(const double u[2], double omega, double out[2])
{
    out[0] =  u[1];
    out[1] = -2.0*ZETA*omega*u[1] - omega*omega*u[0];
}

/*
 * Jacobian A = df/du  (row-major, 2x2)
 *   A = [ 0,           1         ]
 *       [ -omega^2,  -2*zeta*omega ]
 * Stored as A[0..3] = { A00, A01, A10, A11 }.
 * Note: A is state-independent for this linear ODE.
 */
static void df_du_at(const double u[2], double omega, double A[4])
{
    (void)u; /* unused: Jacobian is constant for this linear system */
    A[0] =  0.0;
    A[1] =  1.0;
    A[2] = -omega * omega;
    A[3] = -2.0 * ZETA * omega;
}

/* ================================================================
 * Utility: solve 2x2 linear system  M*x = b  (Cramer's rule)
 * ================================================================ */
static void solve_2x2(const double M[4], const double b[2], double x[2])
{
    double det = M[0]*M[3] - M[1]*M[2];
    if (fabs(det) < 1e-15) {
        /* Near-singular: return b unchanged as a fallback */
        x[0] = b[0]; x[1] = b[1];
        return;
    }
    x[0] = ( M[3]*b[0] - M[1]*b[1]) / det;
    x[1] = (-M[2]*b[0] + M[0]*b[1]) / det;
}

/* ================================================================
 * Forward Euler step
 * ================================================================ */
static void euler_step(const double u[2], double omega, double h,
                       double u_next[2])
{
    double f[2];
    f_rhs(u, omega, f);
    u_next[0] = u[0] + h * f[0];
    u_next[1] = u[1] + h * f[1];
}

/* ================================================================
 * Adjoint gradient w.r.t. initial condition x0
 *
 * Forward pass : explicit Euler
 *   u_{k+1} = u_k + h * f(u_k)        =>  transition matrix J = I + h*A
 *
 * Backward pass: implicit Euler adjoint (symplectic pair)
 *   (I + h*A^T) * lambda_{k-1} = lambda_k
 *
 * Terminal condition: lambda_N = (u_N[0] - x_obs_N, 0)^T
 * Return value      : lambda_0[0]  =  dC/dx0
 * ================================================================ */
static double grad_euler_adjoint(double x0_val, double v0_val,
                                  double omega, double h, int N,
                                  const double *x_obs)
{
    /* --- Forward pass: store all states --- */
    double *U = (double*)malloc(sizeof(double) * 2 * (N + 1));
    U[0] = x0_val;
    U[1] = v0_val;
    for (int k = 0; k < N; k++)
        euler_step(&U[2*k], omega, h, &U[2*(k+1)]);

    /* --- Backward pass: adjoint propagation --- */
    /* Terminal condition: dC/du_N = (x_N - x_obs_N, 0)^T */
    double lam[2] = { U[2*N] - x_obs[N], 0.0 };

    for (int k = N - 1; k >= 0; k--) {
        double A[4];
        df_du_at(&U[2*k], omega, A);

        /*
         * Build M = I + h*A^T
         * A^T swaps off-diagonal: (A^T)_ij = A_ji
         *   M = [ 1 + h*A[0],   h*A[2] ]
         *       [     h*A[1],  1+h*A[3] ]
         */
        double M[4] = {
            1.0 + h*A[0],  h*A[2],
                  h*A[1],  1.0 + h*A[3]
        };

        /* Solve (I + h*A^T) * lambda_{k-1} = lambda_k */
        double l_prev[2];
        solve_2x2(M, lam, l_prev);

        lam[0] = l_prev[0];
        lam[1] = l_prev[1];
    }

    free(U);
    /* lambda_0[0] = dC/dx0,  lambda_0[1] = dC/dv0 */
    return lam[0];
}

/* ================================================================
 * Verification of the symplectic invariant  lambda^T * delta_u = const.
 *
 * Forward Euler transition matrix : J   = I + h*A
 * Adjoint (backward) transition   : J^{-T} applied via solving (I+h*A^T)*lam=lam_next
 *
 * If the adjoint scheme is the correct symplectic pair,
 * the inner product  lambda_k^T * delta_u_k  must be constant for all k.
 * ================================================================ */
static void verify_invariant(double omega, double h, int N,
                              const char *outfile)
{
    /* --- Forward pass --- */
    double *U  = (double*)malloc(sizeof(double) * 2 * (N + 1));
    double *DU = (double*)malloc(sizeof(double) * 2 * (N + 1));

    U[0]  = X0;  U[1]  = V0;
    DU[0] = 1.0; DU[1] = 0.5; /* arbitrary initial variation delta_u0 */

    for (int k = 0; k < N; k++) {
        euler_step(&U[2*k], omega, h, &U[2*(k+1)]);

        /* Propagate variation: delta_u_{k+1} = J * delta_u_k,  J = I + h*A */
        double A[4];
        df_du_at(&U[2*k], omega, A);
        double J[4] = {
            1.0 + h*A[0],       h*A[1],
                  h*A[2],  1.0 + h*A[3]
        };
        DU[2*(k+1)]   = J[0]*DU[2*k] + J[1]*DU[2*k+1];
        DU[2*(k+1)+1] = J[2]*DU[2*k] + J[3]*DU[2*k+1];
    }

    /* --- Backward pass: record invariant at each step --- */
    double lam[2] = { 1.23, 4.56 }; /* arbitrary lambda_N */

    FILE *fp = fopen(outfile, "w");
    fprintf(fp, "# k  lambda_k^T * delta_u_k\n");

    for (int k = N; k >= 0; k--) {
        double inv = lam[0]*DU[2*k] + lam[1]*DU[2*k+1];
        fprintf(fp, "%5d  %.15e\n", k, inv);

        if (k == 0) break;

        /* Adjoint update: solve (I + h*A^T) * lambda_{k-1} = lambda_k */
        double A[4];
        df_du_at(&U[2*(k-1)], omega, A);
        double M[4] = {
            1.0 + h*A[0],  h*A[2],
                  h*A[1],  1.0 + h*A[3]
        };
        double l_prev[2];
        solve_2x2(M, lam, l_prev);
        lam[0] = l_prev[0];
        lam[1] = l_prev[1];
    }

    fclose(fp);
    free(U); free(DU);
    printf("Invariant written to %s\n", outfile);
}

/* ================================================================
 * Output: gradient vs. x0  (fixed omega, varying x0)
 * ================================================================ */
static void write_grad_x0_dat(double omega_fixed, int N)
{
    double h = T_END / N;
    double *x_obs = (double*)malloc(sizeof(double) * (N + 1));
    for (int k = 0; k <= N; k++)
        x_obs[k] = x_exact(k*h, OMEGA);

    FILE *fp = fopen("grad_x0_euler.dat", "w");
    fprintf(fp, "# omega=%.2f  N=%d  h=%.2e\n", omega_fixed, N, h);
    fprintf(fp, "# x0  grad_analytical  grad_euler_adjoint\n");

    int    n_pts  = 200;
    double x0_lo  = -5.0, x0_hi = 5.0;
    for (int i = 0; i < n_pts; i++) {
        double x0 = x0_lo + (x0_hi - x0_lo) * i / (n_pts - 1);
        double gA = (x_exact_with_init(T_END, x0, V0, omega_fixed) - x_obs[N])
                    * dx_dx0_analytical(T_END, omega_fixed);
        double gC = grad_euler_adjoint(x0, V0, omega_fixed, h, N, x_obs);
        fprintf(fp, "%.6f  %+.8e  %+.8e\n", x0, gA, gC);
    }

    fclose(fp);
    free(x_obs);
    printf("grad_x0_euler.dat written\n");
}

/* ================================================================
 * Output: convergence in h  (fixed omega, varying N)
 * ================================================================ */
static void write_converge_dat(double omega_fixed)
{
    static const int Nlist[] = { 50, 100, 200, 500, 1000, 2000, 5000, 10000 };
    int n = (int)(sizeof(Nlist) / sizeof(Nlist[0]));

    FILE *fp = fopen("converge_euler.dat", "w");
    fprintf(fp, "# omega=%.4f  eval at x0=%.4f\n", omega_fixed, X0);
    fprintf(fp, "# h  relerr_euler_adjoint\n");

    printf("\nConvergence test (grad w.r.t. x0, Euler adjoint):\n");
    printf("  %-8s  %-10s  %-14s\n", "N", "h", "rel_err");

    for (int i = 0; i < n; i++) {
        int    N = Nlist[i];
        double h = T_END / N;

        double *x_obs = (double*)malloc(sizeof(double) * (N + 1));
        for (int k = 0; k <= N; k++)
            x_obs[k] = x_exact(k*h, OMEGA);

        double gA  = grad_x0_analytical(omega_fixed, h, N, x_obs);
        double gC  = grad_euler_adjoint(X0, V0, omega_fixed, h, N, x_obs);
        double err = fabs(gC - gA) / (fabs(gA) + 1e-300);

        fprintf(fp, "%.6e  %.8e\n", h, err);
        printf("  %-8d  %-10.2e  %-14.3e\n", N, h, err);

        free(x_obs);
    }
    fclose(fp);
    printf("converge_euler.dat written\n");
}

/* ================================================================
 * main
 * ================================================================ */
int main(void)
{
    write_grad_x0_dat(1.5, 2000);
    write_converge_dat(1.5);
    verify_invariant(1.5, T_END / 1000.0, 1000, "invariant_euler.dat");
    return 0;
}