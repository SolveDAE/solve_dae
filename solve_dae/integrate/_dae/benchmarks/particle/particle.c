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
  N_Vector yy, yp, avtol, y_true, yp_true, w, diff_y, diff_yp;
  sunrealtype rtol, atol, *yval, *ypval, *atval;
  sunrealtype t0, t1, tout, tret;
  int iout, retval, retvalr;
  SUNMatrix A;
  SUNLinearSolver LS;
  SUNNonlinearSolver NLS;
  SUNContext ctx;
  FILE* FID;

  mem = NULL;
  yy = yp = avtol = y_true = yp_true = w = diff_y = NULL;
  yval = ypval = atval = NULL;
  A                    = NULL;
  LS                   = NULL;
  NLS                  = NULL;

  FID = fopen("particle_errors_IDA.csv", "a");
  fprintf(FID, "t, error_y, error_yp\n");
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
    w = N_VClone(yy);
    N_VConst(1.0, w);  // Set all weights to 1.0
    if (check_retval((void*)w, "N_VNew_Serial", 0)) { return (1); }
    diff_y = N_VClone(yy);
    if (check_retval((void*)diff_y, "N_VNew_Serial", 0)) { return (1); }
    diff_yp = N_VClone(yy);
    if (check_retval((void*)diff_yp, "N_VNew_Serial", 0)) { return (1); }

    /* Initialize  y, y' */
    sol_true(t0, yy, yp);
      
    /* define tolerances */
    rtol = pow(10, -(3 + m / 4));
    N_VConst(rtol, avtol);

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

    /* compute error*/
    sol_true(t1, y_true, yp_true);
    N_VLinearSum(1.0, yy, -1.0, y_true, diff_y);
    double error_y = N_VWL2Norm(diff_y, w);
    N_VLinearSum(1.0, yp, -1.0, yp_true, diff_yp);
    double error_yp = N_VWL2Norm(diff_yp, w);

    /* write results to file */
    FID = fopen("particle_errors_IDA.csv", "a");
    fprintf(FID, "%17.17e, %17.17e, %17.17e\n", elapsed_time, error_y, error_y);
    fclose(FID);

    /* Print rtol, elapsed time and error */
    printf("rtol: %e, elapsed time: %e, error_y: %e, error_yp: %e\n", rtol, elapsed_time, error_y, error_yp);

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
  N_VDestroy(w);
  N_VDestroy(diff_y);
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
