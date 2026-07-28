#include <time.h>
#include <ida/ida.h> /* prototypes for IDA fcts., consts.    */
#include <math.h>
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <stdio.h>
#include <sundials/sundials_math.h> /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of sunrealtype, sunindextype      */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to Newton SUNNonlinearSolver  */

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

/* Problem Constants */
#define NEQ  6
#define OMEGA 2 * M_PI

/* Prototypes of functions called by IDA */
int res(sunrealtype tres, N_Vector yy, N_Vector yp, N_Vector resval,
        void* user_data);

/* Prototypes of private functions */
int sol_true(sunrealtype t, N_Vector yy, N_Vector yp);
static int check_retval(void* returnvalue, const char* funcname, int opt);

/*
 *--------------------------------------------------------------------
 * Main Program
 *--------------------------------------------------------------------
 */

int main(void)
{
  void* mem;
  N_Vector yy, yp, avtol, y_true, yp_true;
  sunrealtype rtol, atol, *yval, *ypval, *atval;
  sunrealtype t0, t1, tout, tret;
  int iout, retval, retvalr;
  SUNMatrix A;
  SUNLinearSolver LS;
  SUNNonlinearSolver NLS;
  SUNContext ctx;
  FILE* FID;

  mem = NULL;
  yy = yp = avtol = y_true = yp_true = NULL;
  yval = ypval = atval = NULL;
  A                    = NULL;
  LS                   = NULL;
  NLS                  = NULL;

  /* error_y, error_yp are the naive L2 errors over the full 6-component
   * y and y' vectors (as in the original version of this file) -- kept
   * around because they are still useful for some purposes, e.g.
   * comparing directly against other IDA/DASSL-style benchmarks that
   * report exactly this metric.
   *
   * error_state is the L2 error of just the physical state (x, y, u, v).
   * error_la, error_mu are the errors of the actual Lagrange multipliers
   * lambda, mu, which in this stabilized-index-1 (Anantharaman/Hiller)
   * formulation appear as yp[4], yp[5] (the derivatives of the auxiliary
   * integrated states La, Mu), NOT as yy[4], yy[5]. Reporting them
   * separately from the state matters because the multipliers generally
   * converge at a different (typically lower) order than x, y, u, v, so
   * lumping everything into one combined norm would obscure that and make
   * comparisons against solvers that treat the multipliers differently
   * (e.g. RADAU5's GGL formulation, where lambda, mu are algebraic
   * components of y itself) misleading. */
  /* nstep/naccpt/nrejct/nfev/njev/nlu/nlusolve mirror the metrics exported
   * by the RADAU5 driver (particle_radau5.f90) and by solve_dae's Python
   * solvers (see solve_dae/integrate/_dae/base.py), so all three can be
   * compared directly:
   *   naccpt   = IDAGetNumSteps           (IDA only counts accepted steps)
   *   nrejct   = IDAGetNumErrTestFails + IDAGetNumNonlinSolvConvFails
   *   nstep    = naccpt + nrejct
   *   nfev     = IDAGetNumResEvals
   *   njev     = IDAGetNumJacEvals
   *   nlu      = IDAGetNumLinSolvSetups   (one dense LU factorization per setup)
   *   nlusolve = IDAGetNumNonlinSolvIters (one linear solve per Newton iteration)
   */
  /* "w" (not "a"): each run of this program regenerates the whole file in
   * one go (the loop below writes all m_max+1 rows), so re-running the
   * binary must start from a clean file instead of appending a second
   * header/row block (with a possibly different column count) on top of
   * whatever an older build left behind -- that mismatch is exactly what
   * makes np.loadtxt() in common.py choke with a "number of columns
   * changed" error. */
  FID = fopen("particle_errors_IDA.csv", "w");
  fprintf(FID, "rtol, atol, elapsed_time, error_y, error_yp, error_state, error_la, error_mu, "
               "nstep, naccpt, nrejct, nfev, njev, nlu, nlusolve\n");
  fclose(FID);

  // double m_max = 24.0;
  double m_max = 40.0;
  for (double m=0.0; m<m_max+1.0; m++) {

    /* Integration limits */
    int mxsteps = 1e8;
    t0 = 0.0;
    t1 = 2 * M_PI;

    /* Create SUNDIALS context */
    retval = SUNContext_Create(SUN_COMM_NULL, &ctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) { return (1); }

    /* Allocate N-vectors. */
    yy = N_VNew_Serial(NEQ, ctx);
    if (check_retval((void*)yy, "N_VNew_Serial", 0)) { return (1); }
    yp = N_VClone(yy);
    if (check_retval((void*)yp, "N_VNew_Serial", 0)) { return (1); }
    avtol = N_VClone(yy);
    if (check_retval((void*)avtol, "N_VNew_Serial", 0)) { return (1); }
    y_true = N_VClone(yy);
    if (check_retval((void*)y_true, "N_VNew_Serial", 0)) { return (1); }
    yp_true = N_VClone(yy);
    if (check_retval((void*)yp_true, "N_VNew_Serial", 0)) { return (1); }

    /* Initialize  y, y' */
    sol_true(t0, yy, yp);
      
    /* define tolerances (atol == rtol here, as in particle.py's atols = rtols) */
    rtol = pow(10, -(3 + m / 4));
    atol = rtol;
    N_VConst(atol, avtol);

    /* Call IDACreate and IDAInit to initialize IDA memory */
    mem = IDACreate(ctx);
    if (check_retval((void*)mem, "IDACreate", 0)) { return (1); }
    retval = IDAInit(mem, res, t0, yy, yp);
    if (check_retval(&retval, "IDAInit", 1)) { return (1); }
  
    /* Call IDASVtolerances to set tolerances */
    retval = IDASVtolerances(mem, rtol, avtol);
    if (check_retval(&retval, "IDASVtolerances", 1)) { return (1); }

    /* Create dense SUNMatrix for use in linear solves */
    A = SUNDenseMatrix(NEQ, NEQ, ctx);
    if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return (1); }

    /* Create dense SUNLinearSolver object */
    LS = SUNLinSol_Dense(yy, A, ctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return (1); }

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(mem, LS, A);
    if (check_retval(&retval, "IDASetLinearSolver", 1)) { return (1); }

    /* Create Newton SUNNonlinearSolver object. IDA uses a
    * Newton SUNNonlinearSolver by default, so it is unecessary
    * to create it and attach it. It is done in this example code
    * solely for demonstration purposes. */
    NLS = SUNNonlinSol_Newton(yy, ctx);
    if (check_retval((void*)NLS, "SUNNonlinSol_Newton", 0)) { return (1); }

    /* Attach the nonlinear solver */
    retval = IDASetNonlinearSolver(mem, NLS);
    if (check_retval(&retval, "IDASetNonlinearSolver", 1)) { return (1); }

    /* Maximum number of steps */
    IDASetMaxNumSteps(mem, mxsteps);

    /* Call IDASolve */
    clock_t start = clock();
    retval = IDASolve(mem, t1, &tret, yy, yp, IDA_NORMAL);
    clock_t end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

    /* compute error */
    sol_true(t1, y_true, yp_true);

    sunrealtype* yv      = N_VGetArrayPointer(yy);
    sunrealtype* ypv     = N_VGetArrayPointer(yp);
    sunrealtype* yv_true = N_VGetArrayPointer(y_true);
    sunrealtype* ypv_true = N_VGetArrayPointer(yp_true);

    /* naive errors: L2 norm over the full 6-component y / y' vectors */
    double error_y = 0.0;
    double error_yp = 0.0;
    for (int k = 0; k < NEQ; k++) {
      double dy = yv[k] - yv_true[k];
      double dyp = ypv[k] - ypv_true[k];
      error_y += dy * dy;
      error_yp += dyp * dyp;
    }
    error_y = sqrt(error_y);
    error_yp = sqrt(error_yp);

    /* state error: L2 norm over the physical state (x, y, u, v) only */
    double error_state = 0.0;
    for (int k = 0; k < 4; k++) {
      double d = yv[k] - yv_true[k];
      error_state += d * d;
    }
    error_state = sqrt(error_state);

    /* multiplier errors: lambda = yp[4], mu = yp[5] */
    double error_la = fabs(ypv[4] - ypv_true[4]);
    double error_mu = fabs(ypv[5] - ypv_true[5]);

    /* solver statistics, see comment on the CSV header above */
    long int nsteps = 0, netfails = 0, nncfails = 0;
    long int nrevals = 0, njevals = 0, nlinsetups = 0, nniters = 0;
    IDAGetNumSteps(mem, &nsteps);
    IDAGetNumErrTestFails(mem, &netfails);
    IDAGetNumNonlinSolvConvFails(mem, &nncfails);
    IDAGetNumResEvals(mem, &nrevals);
    IDAGetNumJacEvals(mem, &njevals);
    IDAGetNumLinSolvSetups(mem, &nlinsetups);
    IDAGetNumNonlinSolvIters(mem, &nniters);

    long int naccpt = nsteps;
    long int nrejct = netfails + nncfails;
    long int nstep  = naccpt + nrejct;
    long int nfev   = nrevals;
    long int njev   = njevals;
    long int nlu    = nlinsetups;
    long int nlusolve = nniters;

    /* write results to file. Float columns use %.4e (lowercase e, matching
     * the fmt=["%.4e"]*n_float_cols+["%d"]*n_stats used by common.py's
     * Python export) so all three benchmark drivers produce the same
     * number format; the stat columns are plain integers (%ld), never
     * scientific notation. */
    FID = fopen("particle_errors_IDA.csv", "a");
    fprintf(FID, "%.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, "
                 "%ld, %ld, %ld, %ld, %ld, %ld, %ld\n",
            rtol, atol, elapsed_time, error_y, error_yp, error_state, error_la, error_mu,
            nstep, naccpt, nrejct, nfev, njev, nlu, nlusolve);
    fclose(FID);

    /* Print rtol, atol, elapsed time and errors */
    printf("rtol: %e, atol: %e, elapsed time: %e, error_y: %e, error_yp: %e, "
           "error_state: %e, error_la: %e, error_mu: %e, "
           "nstep: %ld, naccpt: %ld, nrejct: %ld, nfev: %ld, njev: %ld, nlu: %ld, nlusolve: %ld\n",
           rtol, atol, elapsed_time, error_y, error_yp, error_state, error_la, error_mu,
           nstep, naccpt, nrejct, nfev, njev, nlu, nlusolve);

  }

  /* Free memory */
  IDAFree(&mem);
  SUNNonlinSolFree(NLS);
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(yy);
  N_VDestroy(yp);
  N_VDestroy(avtol);
  N_VDestroy(y_true);
  N_VDestroy(yp_true);
  SUNContext_Free(&ctx);

  return (retval);
}

/*
 *--------------------------------------------------------------------
 * Helper functions
 *--------------------------------------------------------------------
 */
// The time derivative of this function has to be phi_dot(t)**2."
sunrealtype PHI(sunrealtype t)
{
  return OMEGA * OMEGA * (t / 2.0 + sin(2 * t) / 4.0);
}

sunrealtype phi(sunrealtype t)
{
  return OMEGA * sin(t);
}

sunrealtype phi_p(sunrealtype t)
{
  return OMEGA * cos(t);
}

sunrealtype phi_pp(sunrealtype t)
{
  return -OMEGA * sin(t);
}

/*
 *--------------------------------------------------------------------
 * Functions called by IDA
 *--------------------------------------------------------------------
 */

/*
 * Define the system residual function.
 */
int res(sunrealtype t, N_Vector yy, N_Vector yp, N_Vector rr,
           void* user_data)
{
  sunrealtype *yval, *ypval, *rval;

  yval  = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);
  rval  = N_VGetArrayPointer(rr);

  sunrealtype force = phi_pp(t);

  sunrealtype x = yval[0];
  sunrealtype y = yval[1];
  sunrealtype u = yval[2];
  sunrealtype v = yval[3];

  sunrealtype x_dot = ypval[0];
  sunrealtype y_dot = ypval[1];
  sunrealtype u_dot = ypval[2];
  sunrealtype v_dot = ypval[3];
  sunrealtype Lap = ypval[4];
  sunrealtype Mup = ypval[5];

  rval[0] = x_dot - (u + 2 * x * Mup);
  rval[1] = y_dot - (v + 2 * y * Mup);
  rval[2] = u_dot - (2 * x * Lap - y * force);
  rval[3] = v_dot - (2 * y * Lap + x * force);
  rval[4] = 2 * (x * u + y * v);
  rval[5] = x * x + y * y - 1.0;

  return (0);
}

/*
 *--------------------------------------------------------------------
 * Private functions
 *--------------------------------------------------------------------
 */

int sol_true(sunrealtype t, N_Vector yy, N_Vector yp)
{
  sunrealtype *yval, *ypval;

  sunrealtype t2 = t * t;

  yval  = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);

  sunrealtype phi_val = phi(t);
  sunrealtype sin_phi = sin(phi_val);
  sunrealtype cos_phi = cos(phi_val);
  sunrealtype phi_p_val = phi_p(t);
  sunrealtype phi_pp_val = phi_pp(t);

  yval[0] = cos_phi;
  yval[1] = sin_phi;
  yval[2] = -sin_phi * phi_p_val;
  yval[3] = cos_phi * phi_p_val;
  yval[4] = -PHI(t) / 2;
  yval[5] = 0;

  ypval[0] = -sin_phi * phi_p_val;
  ypval[1] = cos_phi * phi_p_val;
  ypval[2] = -cos_phi * phi_p_val * phi_p_val - sin_phi * phi_pp_val;
  ypval[3] = -sin_phi * phi_p_val * phi_p_val + cos_phi * phi_pp_val;
  ypval[4] = -phi_p_val * phi_p_val / 2;
  ypval[5] = 0;

  return (0);
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

static int check_retval(void* returnvalue, const char* funcname, int opt)
{
  int* retval;
  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }
  else if (opt == 1)
  {
    /* Check if retval < 0 */
    retval = (int*)returnvalue;
    if (*retval < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return (1);
    }
  }
  else if (opt == 2 && returnvalue == NULL)
  {
    /* Check if function returned NULL pointer - no memory allocated */
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  return (0);
}
