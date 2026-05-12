/* ================================================================
 * 1d_init_heun.c
 *
 * Gradient computation w.r.t. initial conditions (x0, v0)
 * for a 1-D damped oscillator via the symplectic implicit adjoint
 * method paired with the explicit Heun (RK2) forward scheme.
 *
 * ODE:  u' = f(u),  u = (x, v)
 *   f[0] =  v
 *   f[1] = -2*zeta*omega*v - omega^2*x
 *
 * Cost:  C = (1/2) * (x_N - x_obs_N)^2   (terminal cost only)
 *
 * Adjoint scheme: grad_symplectic_implicit
 * ---------------------------------------------------------------
 * Heun forward transition matrix (Jacobian of the map u_k -> u_{k+1}):
 *
 *   u*       = u_k + h * f(u_k)          (Euler predictor stage)
 *   u_{k+1}  = u_k + (h/2)*(f(u_k) + f(u*))
 *
 *   J = I + (h/2)*(A1 + A2) + (h^2/2)*(A2*A1)
 *   where A1 = df/du(u_k),  A2 = df/du(u*)
 *
 * Symplectic adjoint update solves the transpose system:
 *   M * lambda_{k-1} = lambda_k
 *   M = I - (h/2)*(A1^T + A2^T) + (h^2/2)*(A1^T * A2^T)
 *     = J^{-T}  (exact for linear systems)
 *
 * This ensures  lambda_k^T * delta_u_k = const.  at every step.
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
 * Heun (explicit RK2) forward step
 * ================================================================ */
static void heun_step(const double u[2], double omega, double h,
                      double u_next[2])
{
    double s1[2], s2[2], tmp[2];
    f_rhs(u, omega, s1);
    tmp[0] = u[0] + h*s1[0];
    tmp[1] = u[1] + h*s1[1];
    f_rhs(tmp, omega, s2);
    u_next[0] = u[0] + 0.5*h*(s1[0] + s2[0]);
    u_next[1] = u[1] + 0.5*h*(s1[1] + s2[1]);
}

/*
 * Compute the Heun transition matrix J and its transpose J^T
 * at state u_k with step size h.
 *
 * Forward map:
 *   u* = u_k + h*f(u_k)
 *   u_{k+1} = u_k + (h/2)*(f(u_k) + f(u*))
 *
 * Transition matrix (linearisation of the above):
 *   J = I + (h/2)*(A1 + A2) + (h^2/2)*(A2*A1)
 *
 * Transpose J^T:
 *   JT[i][j] = J[j][i]
 *
 * Indices (row-major): M[0]=M00, M[1]=M01, M[2]=M10, M[3]=M11
 */
static void heun_transition(const double uk[2], double omega, double h,
                             double J[4], double JT[4])
{
    double f1[2], u_star[2], A1[4], A2[4];
    f_rhs(uk, omega, f1);
    u_star[0] = uk[0] + h*f1[0];
    u_star[1] = uk[1] + h*f1[1];
    df_du_at(uk,     omega, A1);
    df_du_at(u_star, omega, A2);

    double h2 = 0.5*h, h2_2 = 0.5*h*h;

    /* J = I + (h/2)*(A1+A2) + (h^2/2)*(A2*A1) */
    J[0] = 1.0 + h2*(A1[0]+A2[0]) + h2_2*(A2[0]*A1[0] + A2[1]*A1[2]);
    J[1] =       h2*(A1[1]+A2[1]) + h2_2*(A2[0]*A1[1] + A2[1]*A1[3]);
    J[2] =       h2*(A1[2]+A2[2]) + h2_2*(A2[2]*A1[0] + A2[3]*A1[2]);
    J[3] = 1.0 + h2*(A1[3]+A2[3]) + h2_2*(A2[2]*A1[1] + A2[3]*A1[3]);

    /* J^T: swap off-diagonal entries */
    JT[0] = J[0]; JT[1] = J[2];
    JT[2] = J[1]; JT[3] = J[3];
}

/* ================================================================
 * Symplectic implicit adjoint gradient w.r.t. initial condition x0
 *
 * Forward pass : explicit Heun (RK2)
 * Backward pass: solve  M * lambda_{k-1} = lambda_k  at each step
 *
 *   M = I - (h/2)*(A1^T + A2^T) + (h^2/2)*(A1^T * A2^T)
 *
 * For this linear ODE, M = J^{-T} exactly, so the invariant
 *   lambda_k^T * delta_u_k = const.
 * holds to machine precision at every step.
 *
 * Terminal condition: lambda_N = (u_N[0] - x_obs_N, 0)^T
 * Return value      : lambda_0[0]  =  dC/dx0
 * ================================================================ */
static double grad_symplectic_implicit(double x0_val, double v0_val,
                                        double omega, double h, int N,
                                        const double *x_obs)
{
    /* --- Forward pass: store all states --- */
    double *U = (double*)malloc(sizeof(double) * 2 * (N + 1));
    U[0] = x0_val;
    U[1] = v0_val;
    for (int k = 0; k < N; k++)
        heun_step(&U[2*k], omega, h, &U[2*(k+1)]);

    /* --- Backward pass --- */
    /* Terminal condition: dC/du_N = (x_N - x_obs_N, 0)^T */
    double lam[2] = { U[2*N] - x_obs[N], 0.0 };

    for (int k = N - 1; k >= 0; k--) {
        double f1[2], u_star[2], A1[4], A2[4];
        const double *uk = &U[2*k];
        f_rhs(uk, omega, f1);
        u_star[0] = uk[0] + h*f1[0];
        u_star[1] = uk[1] + h*f1[1];
        df_du_at(uk,     omega, A1);
        df_du_at(u_star, omega, A2);

        /*
         * Symplectic adjoint: apply J^T explicitly.
         *
         * J  = I + (h/2)*(A1+A2)     + (h^2/2)*(A2*A1)
         * J^T= I + (h/2)*(A1^T+A2^T) + (h^2/2)*(A1^T*A2^T)
         *
         * A^T row-major: (A^T)_00=A[0], (A^T)_01=A[2], (A^T)_10=A[1], (A^T)_11=A[3]
         */
        double h2 = 0.5*h, h2_2 = 0.5*h*h;
        double JT[4];
        JT[0] = 1.0 + h2*(A1[0]+A2[0]) + h2_2*(A1[0]*A2[0] + A1[2]*A2[1]);
        JT[1] =       h2*(A1[2]+A2[2]) + h2_2*(A1[0]*A2[2] + A1[2]*A2[3]);
        JT[2] =       h2*(A1[1]+A2[1]) + h2_2*(A1[1]*A2[0] + A1[3]*A2[1]);
        JT[3] = 1.0 + h2*(A1[3]+A2[3]) + h2_2*(A1[1]*A2[2] + A1[3]*A2[3]);

        double l_prev[2];
        l_prev[0] = JT[0]*lam[0] + JT[1]*lam[1];
        l_prev[1] = JT[2]*lam[0] + JT[3]*lam[1];
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
 * Forward variation propagation:
 *   delta_u_{k+1} = J * delta_u_k
 *
 * Adjoint propagation (symplectic implicit):
 *   M * lambda_{k-1} = lambda_k,  M = J^{-T}
 *
 * Hence:
 *   lambda_{k-1}^T * delta_u_{k-1}
 *     = (M^{-T} lambda_k)^T * (J^{-1} delta_u_k ... wait, no:
 *   = lambda_k^T * M^{-1} * J * delta_u_k  ... (*)
 *
 * For a linear ODE, M = J^{-T} exactly, so M^{-1} = J^T and
 *   (*) = lambda_k^T * J^T * J * ... no — more directly:
 *   lambda_k^T * delta_u_k
 *     = (J^T lambda_{k-1})^T * delta_u_k
 *     = lambda_{k-1}^T * J * delta_u_k
 *     = lambda_{k-1}^T * delta_u_{k-1}       (QED)
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
        heun_step(&U[2*k], omega, h, &U[2*(k+1)]);

        /* Propagate variation: delta_u_{k+1} = J * delta_u_k */
        double J[4], JT[4];
        heun_transition(&U[2*k], omega, h, J, JT);
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

        /* Adjoint update: solve M * lambda_{k-1} = lambda_k */
        double f1[2], u_star[2], A1[4], A2[4];
        const double *uk = &U[2*(k-1)];
        f_rhs(uk, omega, f1);
        u_star[0] = uk[0] + h*f1[0];
        u_star[1] = uk[1] + h*f1[1];
        df_du_at(uk,     omega, A1);
        df_du_at(u_star, omega, A2);

        /* Adjoint update: lambda_{k-1} = J^T * lambda_k */
        double h2 = 0.5*h, h2_2 = 0.5*h*h;
        double JT[4];
        JT[0] = 1.0 + h2*(A1[0]+A2[0]) + h2_2*(A1[0]*A2[0] + A1[2]*A2[1]);
        JT[1] =       h2*(A1[2]+A2[2]) + h2_2*(A1[0]*A2[2] + A1[2]*A2[3]);
        JT[2] =       h2*(A1[1]+A2[1]) + h2_2*(A1[1]*A2[0] + A1[3]*A2[1]);
        JT[3] = 1.0 + h2*(A1[3]+A2[3]) + h2_2*(A1[1]*A2[2] + A1[3]*A2[3]);

        double l_prev[2];
        l_prev[0] = JT[0]*lam[0] + JT[1]*lam[1];
        l_prev[1] = JT[2]*lam[0] + JT[3]*lam[1];
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
        // x_obs[k] = x_exact(k*h, OMEGA);
        x_obs[k] = x_exact_with_init(k*h, 1.5, V0, OMEGA);

    FILE *fp = fopen("grad_x0_heun.dat", "w");
    fprintf(fp, "# omega=%.2f  N=%d  h=%.2e\n", omega_fixed, N, h);
    fprintf(fp, "# x0  grad_analytical  grad_heun_adjoint\n");

    int    n_pts = 200;
    double x0_lo = -5.0, x0_hi = 5.0;
    for (int i = 0; i < n_pts; i++) {
        double x0 = x0_lo + (x0_hi - x0_lo) * i / (n_pts - 1);
        double gA = (x_exact_with_init(T_END, x0, V0, omega_fixed) - x_obs[N])
                    * dx_dx0_analytical(T_END, omega_fixed);
        double gC = grad_symplectic_implicit(x0, V0, omega_fixed, h, N, x_obs);
        fprintf(fp, "%.6f  %+.8e  %+.8e\n", x0, gA, gC);
    }

    fclose(fp);
    free(x_obs);
    printf("grad_x0_heun.dat written\n");
}

/* ================================================================
 * Output: convergence in h  (fixed omega, varying N)
 * ================================================================ */
static void write_converge_dat(double omega_fixed)
{
    static const int Nlist[] = { 50, 100, 200, 500, 1000, 2000, 5000, 10000 };
    int n = (int)(sizeof(Nlist) / sizeof(Nlist[0]));

    FILE *fp = fopen("converge_heun.dat", "w");
    fprintf(fp, "# omega=%.4f  eval at x0=%.4f\n", omega_fixed, X0);
    fprintf(fp, "# h  relerr_symplectic_adjoint\n");

    printf("\nConvergence test (grad w.r.t. x0, Heun adjoint):\n");
    printf("  %-8s  %-10s  %-14s\n", "N", "h", "rel_err");

    for (int i = 0; i < n; i++) {
        int    N = Nlist[i];
        double h = T_END / N;

        double *x_obs = (double*)malloc(sizeof(double) * (N + 1));
        for (int k = 0; k <= N; k++)
            x_obs[k] = x_exact(k*h, OMEGA);

        double gA  = grad_x0_analytical(omega_fixed, h, N, x_obs);
        double gC  = grad_symplectic_implicit(X0, V0, omega_fixed, h, N, x_obs);
        double err = fabs(gC - gA) / (fabs(gA) + 1e-300);

        fprintf(fp, "%.6e  %.8e\n", h, err);
        printf("  %-8d  %-10.2e  %-14.3e\n", N, h, err);

        free(x_obs);
    }
    fclose(fp);
    printf("converge_heun.dat written\n");
}

/* ================================================================
 * main
 * ================================================================ */
int main(void)
{
    write_grad_x0_dat(OMEGA, 2000);
    write_converge_dat(1.5);
    verify_invariant(1.5, T_END / 1000.0, 1000, "invariant_heun.dat");
    return 0;
}