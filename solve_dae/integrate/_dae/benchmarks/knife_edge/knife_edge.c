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
#define NEQ  7

#define pi acos(-1)
#define m 1.25
#define Theta 0.13
#define g 9.81
#define Omega (pi / 3.0)
#define alpha (pi / 4.0)
#define salpha sin(alpha)

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
  N_Vector yy, yp, avtol, y_true, yp_true, w, diff;
  sunrealtype rtol, *yval, *ypval, *atval;
  sunrealtype t0, t1, tout, tret;
  int iout, retval, retvalr;
  SUNMatrix A;
  SUNLinearSolver LS;
  SUNNonlinearSolver NLS;
  SUNContext ctx;
  FILE* FID;

  mem = NULL;
  yy = yp = avtol = y_true = yp_true = w = diff = NULL;
  yval = ypval = atval = NULL;
  A                    = NULL;
  LS                   = NULL;
  NLS                  = NULL;

  double m_max = 32.0;
  for (double mm=0.0; mm<m_max+1.0; mm++) {

    /* Integration limits */
    int mxsteps = 1e8;
    t0 = 0.0;
    t1 = (2.0 * pi) / Omega;

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
    diff = N_VClone(yy);
    if (check_retval((void*)diff, "N_VNew_Serial", 0)) { return (1); }

    /* Initialize  y, y' */
    sol_true(t0, yy, yp);
    
    /* define tolerances */
    rtol = pow(10, -(1 + mm / 4));
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
    N_VLinearSum(1.0, yy, -1.0, y_true, diff);
    double error = N_VWL2Norm(diff, w);
  
    /* write results to file */
    FID = fopen("knife_edge_errors_IDA.csv", "a");
    fprintf(FID, "%17.17e, %17.17e\n", error, elapsed_time);
    fclose(FID);

    /* Print rtol, elapsed time and error */
    printf("rtol: %e, elapsed time: %e, error: %e\n", rtol, elapsed_time, error);

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
  N_VDestroy(diff);
  SUNContext_Free(&ctx);

  return (retval);
}

/*
 *--------------------------------------------------------------------
 * Functions called by IDA
 *--------------------------------------------------------------------
 */

/*
 * Define the system residual function.
 */
int res(sunrealtype tres, N_Vector yy, N_Vector yp, N_Vector rr,
           void* user_data)
{
  sunrealtype *yval, *ypval, *rval;

  yval  = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);
  rval  = N_VGetArrayPointer(rr);

  sunrealtype x = yval[0];
  sunrealtype y = yval[1];
  sunrealtype phi = yval[2];
  sunrealtype u = yval[3];
  sunrealtype v = yval[4];
  sunrealtype omega = yval[5];
  sunrealtype La = yval[6];

  sunrealtype x_dot = ypval[0];
  sunrealtype y_dot = ypval[1];
  sunrealtype phi_dot = ypval[2];
  sunrealtype u_dot = ypval[3];
  sunrealtype v_dot = ypval[4];
  sunrealtype omega_dot = ypval[5];
  sunrealtype Lap = ypval[6];

  sunrealtype sphi = sin(phi);
  sunrealtype cphi = cos(phi);

  rval[0] = x_dot - u;
  rval[1] = y_dot - v;
  rval[2] = phi_dot - omega;
  rval[3] = m * u_dot - m * g * salpha - sphi * Lap;
  rval[4] = m * v_dot + cphi * Lap;
  rval[5] = Theta * omega_dot;
  rval[6] = u * sphi - v * cphi;

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

  sunrealtype x = (g * salpha / (2.0 * pow(Omega, 2.0))) * pow(sin(Omega * t), 2.0);
  sunrealtype y = (g * salpha / (2.0 * pow(Omega, 2.0))) * (Omega * t - 0.5 * sin(2 * Omega * t));
  sunrealtype phi = Omega * t;
  
  sunrealtype u =  (g * salpha / Omega) * sin(Omega * t) * cos(Omega * t);
  sunrealtype v = (g * salpha / Omega) * pow(sin(Omega * t), 2.0);
  sunrealtype omega = Omega;
  
  sunrealtype La = (2.0 * m * g * salpha / Omega) * (cos(Omega * t) - 1.0);
  sunrealtype La_dot = -2.0 * m * g * salpha * sin(Omega * t);

  sunrealtype x_dot = u;
  sunrealtype y_dot = v;
  sunrealtype phi_dot = omega;

  sunrealtype u_dot = g * salpha + sin(Omega * t) * La_dot / m;
  sunrealtype v_dot = -cos(Omega * t) * La_dot / m;
  sunrealtype omega_dot = 0;

  yval  = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);

  yval[0] = x;
  yval[1] = y;
  yval[2] = phi;
  yval[3] = u;
  yval[4] = v;
  yval[5] = omega;
  yval[6] = La;

  ypval[0] = x_dot;
  ypval[1] = y_dot;
  ypval[2] = phi_dot;
  ypval[3] = u_dot;
  ypval[4] = v_dot;
  ypval[5] = omega_dot;
  ypval[6] = La_dot;

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
