/***********************************************************************
* Adaptive Simulated Annealing (ASA)
* Lester Ingber <ingber@ingber.com>
* Copyright (c) 1987-2013 Lester Ingber.  All Rights Reserved.
* ASA-LICENSE file has the license that must be included with ASA code.
***********************************************************************/

#define USER_ID "/* $Id: asa_usr.c,v 29.6 2013/10/19 21:30:59 ingber Exp ingber $ */"

#include<time.h>
#include "papi.h"
#include "asa_usr.h"
char* DESIGN;
double ERROR_THRESH;
int TargetIter;
asm_stuff stuff;
HOST(opcode,int, MAX_INSTRUCTIONS);
HOST(src0,int, MAX_INSTRUCTIONS);
HOST(src1,int, MAX_INSTRUCTIONS);
HOST(dest,int, MAX_INSTRUCTIONS);
HOST(in_lo,REAL, MAX_INPUTS);
HOST(in_hi,REAL, MAX_INPUTS);

using namespace std;
using namespace thrust;

#if MY_TEMPLATE                 /* MY_TEMPLATE_includes */
  /* add your own include files as required */
#endif

#if ASA_LIB
static LONG_INT *asa_rand_seed;
#endif

#if ASA_SAVE
static double random_array[SHUFFLE];
#endif

#if ASA_FUZZY
static double **FuzzyParameters;
static double *FuzzyValues, *FuzzyMinima, *auxV;
static double ValMinLoc;
#endif

#if SELF_OPTIMIZE
static LONG_INT funevals = 0;
#else

char user_exit_msg[160];        /* temp storage for exit messages */
FILE *ptr_out;

/***********************************************************************
* main
*	This is a sample calling program to optimize using ASA
***********************************************************************/
#if HAVE_ANSI

#if ASA_LIB
int
asa_main (
#if ASA_TEMPLATE_LIB
           double *main_cost_value,
           double *main_cost_parameters, int *main_exit_code
#endif                          /* ASA_TEMPLATE_LIB */
#if ASA_TEMPLATE
#if OPTIONAL_DATA_PTR
           /* insert "," if previous parameters */
           OPTIONAL_PTR_TYPE * OptionalPointer
#endif                          /* OPTIONAL_DATA_PTR */
#endif                          /* ASA_TEMPLATE */
  )
#else /* ASA_LIB */
int
main (int argc, char **argv)
#endif                          /* ASA_LIB */
#else /* HAVE_ANSI */

#if ASA_LIB
int
asa_main (
#if ASA_TEMPLATE_LIB
           main_cost_value, main_cost_parameters, main_exit_code
#endif                          /* ASA_TEMPLATE_LIB */
#if ASA_TEMPLATE
           /* insert "," if previous parameters */
           OptionalPointer
#endif                          /* ASA_TEMPLATE */
  )
#if ASA_TEMPLATE_LIB
     double *main_cost_value;
     double *main_cost_parameters;
     int *main_exit_code;
#endif /* ASA_TEMPLATE_LIB */
#if ASA_TEMPLATE
     OPTIONAL_PTR_TYPE *OptionalPointer;
#endif /* ASA_TEMPLATE */
#else /* ASA_LIB */
int
main (argc, argv)
     int argc;
     char **argv;
#endif /* ASA_LIB */

#endif /* HAVE_ANSI */
{
    long long start_time, end_time;
    PAPI_library_init(PAPI_VER_CURRENT);
    start_time = PAPI_get_real_usec();

  int *exit_code;
  ALLOC_INT n_param;
#if ASA_LIB
#else
  int compile_cnt;
#endif
#if ASA_TEMPLATE_SAMPLE
  FILE *ptr_asa;
#endif
#if ASA_TEMPLATE_ASA_OUT_PID
  char pid_file[18];
  int pid_int;
#endif
#if MULTI_MIN
  int multi_index;
#endif

  /* benchmark name*/
  DESIGN = argv[1];
  char filename[100];
  sprintf(filename, "%s", DESIGN);
  stuff = parse_asm(filename, &hv_opcode, &hv_src0, &hv_src1, &hv_in_lo, &hv_in_hi, &hv_dest);
  /* user-specified error */
  ERROR_THRESH = atof(argv[2]);
  /* number of iterations after which force ASA to stop */
  TargetIter = atoi(argv[3]);
//  printf("TargetIter=%d\n",TargetIter);

  /* pointer to array storage for asa arguments */
  double *parameter_lower_bound, *parameter_upper_bound, *cost_parameters,
    *cost_tangents, *cost_curvature;
  double cost_value;

  int initialize_parameters_value;
  static int number_asa_usr_open = 0;

  /* the number of parameters to optimize */
  ALLOC_INT *parameter_dimension;

  /* pointer to array storage for parameter type flags */
  int *parameter_int_real;

  /* valid flag for cost function */
  int *cost_flag;

  /* seed for random number generator */
  LONG_INT *rand_seed;

  USER_DEFINES *USER_OPTIONS;

#if OPTIONS_FILE
  int fscanf_ret;
  FILE *ptr_options;
  char read_option[80];
  char read_if[4], read_FALSE[6], read_comm1[3], read_ASA_SAVE[9],
    read_comm2[3];
  int read_int;
#if INT_LONG
  LONG_INT read_long;
#endif
  double read_double;
#endif /* OPTIONS_FILE */
#if MY_TEMPLATE                 /* MY_TEMPLATE_main_decl */
  /* add some declarations if required */
#endif

  fscanf_ret = 0;               /* stop compiler warning */
  if (fscanf_ret) {
    ;
  }
#if ASA_TEMPLATE_MULTIPLE
  int n_asa, n_trajectory;
  ALLOC_INT index;
#if HAVE_ANSI
  char asa_file[8] = "asa_x_y";
#else
  char asa_file[8];
#endif /* HAVE_ANSI */
#endif /* ASA_TEMPLATE_MULTIPLE */

#if ASA_TEMPLATE_MULTIPLE
#if HAVE_ANSI
#else
  asa_file[0] = asa_file[2] = 'a';
  asa_file[1] = 's';
  asa_file[3] = asa_file[5] = '_';
  asa_file[4] = 'x';
  asa_file[6] = 'y';
  asa_file[7] = '\0';
#endif /* HAVE_ANSI */
#endif /* ASA_TEMPLATE_MULTIPLE */

  if ((USER_OPTIONS =
       (USER_DEFINES *) calloc (1, sizeof (USER_DEFINES))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): USER_DEFINES");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if OPTIONAL_DATA_PTR
#if ASA_TEMPLATE
  /* see note at "Instead of freeing Asa_Data_Ptr" */
  /* USER_OPTIONS->Asa_Data_Dim_Ptr = 1; */
  /* USER_OPTIONS->Asa_Data_Ptr = OptionalPointer; */
  USER_OPTIONS->Asa_Data_Dim_Ptr = 256;
  if ((USER_OPTIONS->Asa_Data_Ptr =
       (OPTIONAL_PTR_TYPE *) calloc (USER_OPTIONS->Asa_Data_Dim_Ptr,
                                     sizeof (OPTIONAL_PTR_TYPE))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): USER_OPTIONS->Asa_Data_Ptr");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#endif /* ASA_TEMPLATE */
#endif /* OPTIONAL_DATA_PTR */

#if ASA_TEMPLATE_ASA_OUT_PID
  pid_file[0] = 'a';
  pid_file[1] = 's';
  pid_file[2] = 'a';
  pid_file[3] = '_';
  pid_file[4] = 'u';
  pid_file[5] = 's';
  pid_file[6] = 'r';
  pid_file[7] = '_';
  pid_file[8] = 'o';
  pid_file[9] = 'u';
  pid_file[10] = 't';
  pid_file[11] = '_';
  pid_file[17] = '\0';

  pid_int = getpid ();
  if (pid_int < 0) {
    pid_int = -pid_int;
  }

  if (pid_int > 99999) {
    pid_file[11] = '1';
    pid_int = pid_int % 100000;
  }

  if (pid_int < 10 && pid_int > 0) {
    pid_file[12] = '0';
    pid_file[13] = '0';
    pid_file[14] = '0';
    pid_file[15] = '0';
    pid_file[16] = '0' + pid_int;
  } else if (pid_int >= 10 && pid_int < 100) {
    pid_file[12] = '0';
    pid_file[13] = '0';
    pid_file[14] = '0';
    pid_file[15] = '0' + (int) (pid_int / 10);
    pid_file[16] = '0' + (pid_int % 10);
  } else if (pid_int >= 100 && pid_int < 1000) {
    pid_file[12] = '0';
    pid_file[13] = '0';
    pid_file[14] = '0' + (int) (pid_int / 100);
    pid_file[15] = '0' + (int) ((pid_int % 100) / 10);
    pid_file[16] = '0' + ((pid_int % 100) % 10);
  } else if (pid_int >= 1000 && pid_int < 10000) {
    pid_file[12] = '0';
    pid_file[13] = '0' + (int) (pid_int / 1000);
    pid_file[14] = '0' + (int) ((pid_int % 1000) / 100);
    pid_file[15] = '0' + (int) (((pid_int % 1000) % 100) / 10);
    pid_file[16] = '0' + (((pid_int % 1000) % 100) % 10);
  } else if (pid_int >= 10000 && pid_int <= 99999) {
    pid_file[12] = '0' + (int) (pid_int / 10000);
    pid_file[13] = '0' + (int) ((pid_int % 10000) / 1000);
    pid_file[14] = '0' + (int) (((pid_int % 10000) % 1000) / 100);
    pid_file[15] = '0' + (int) (((pid_int % 10000) % 1000) % 100 / 10);
    pid_file[16] = '0' + ((((pid_int % 10000) % 1000) % 100) % 10);
  } else {
    pid_file[11] = '0';
    pid_file[12] = '0';
    pid_file[13] = '0';
    pid_file[14] = '0';
    pid_file[15] = '0';
    pid_file[16] = '0';
  }
  ptr_out = fopen (pid_file, "w");
#else /* ASA_TEMPLATE_ASA_OUT_PID */

  ++number_asa_usr_open;
  /* open the output file */
  /* set "w" to "a" to save data from multiple runs */
#if ASA_SAVE
  if (!strcmp (USER_OUT, "STDOUT")) {
#if INCL_STDOUT
    ptr_out = stdout;
#endif /* INCL_STDOUT */
  } else {
    ptr_out = fopen (USER_OUT, "a");
  }
#else /* ASA_SAVE */
  if (!strcmp (USER_OUT, "STDOUT")) {
#if INCL_STDOUT
    ptr_out = stdout;
#endif /* INCL_STDOUT */
  } else {
#if USER_ASA_USR_OUT
    ;
#else
    ptr_out = fopen (USER_OUT, "w");
#if ASA_TEMPLATE
    /* if multiple calls are to be saved */
    ptr_out = fopen (USER_OUT, "a");
#endif /* ASA_TEMPLATE */
#endif /* USER_ASA_USR_OUT */
  }
#endif /* ASA_SAVE */
#if INCL_STDOUT
  /* use this instead if you want output to stdout */
#endif /* INCL_STDOUT */
#if FALSE
#if INCL_STDOUT
  ptr_out = stdout;
#endif /* INCL_STDOUT */
#endif

#if USER_ASA_USR_OUT
  if ((USER_OPTIONS->Asa_Usr_Out_File =
       (char *) calloc (80, sizeof (char))) == NULL) {
    strcpy (user_exit_msg,
            "main()/asa_main(): USER_OPTIONS->Asa_Usr_Out_File");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  sprintf (USER_OPTIONS->Asa_Usr_Out_File, "%s", "asa_usr_out_my");

  /* OR create memory for file asa_usr_out_my */
  /* which must be "free(asa_usr_out_my);" after use */
  strcpy (USER_OPTIONS->Asa_Usr_Out_File, asa_usr_out_my);

  ptr_out = fopen (USER_OPTIONS->Asa_Usr_Out_File, "w");
#endif /* ASA_TEMPLATE */
#endif /* USER_ASA_USR_OUT */

#endif /* ASA_TEMPLATE_ASA_OUT_PID */
  fprintf (ptr_out, "%s\n\n", USER_ID);
  if (number_asa_usr_open > 1) {
    fprintf (ptr_out, "\n\n\t\t number_asa_usr_open = %d\n",
             number_asa_usr_open);
    fflush (ptr_out);
  }
#if ASA_LIB
#else
  /* print out compile options set by user in Makefile */
  if (argc > 1) {
    fprintf (ptr_out, "CC = %s\n", argv[1]);
    for (compile_cnt = 2; compile_cnt < argc; ++compile_cnt) {
      fprintf (ptr_out, "\t%s\n", argv[compile_cnt]);
    }
    fprintf (ptr_out, "\n");
  }
#endif
#if TIME_CALC
  /* print starting time */
  print_time ("start", ptr_out);
#endif
  fflush (ptr_out);

  if ((rand_seed = (ALLOC_INT *) calloc (1, sizeof (ALLOC_INT))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): rand_seed");
    Exit_USER (user_exit_msg);
    free (USER_OPTIONS);
    return (-2);
  }
#if ASA_LIB
  *rand_seed = (asa_rand_seed ? *asa_rand_seed : (LONG_INT) 696969);
#else
  *rand_seed = 696969;
#endif

  /* initialize random number generator with first call */
  resettable_randflt (rand_seed, 1);

  /* Initialize the users parameters, allocating space, etc.
     Note that the default is to have asa generate the initial
     cost_parameters that satisfy the user's constraints. */

  if ((parameter_dimension =
       (ALLOC_INT *) calloc (1, sizeof (ALLOC_INT))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): parameter_dimension");
    Exit_USER (user_exit_msg);
    free (USER_OPTIONS);
    return (-2);
  }
  if ((exit_code = (int *) calloc (1, sizeof (int))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): exit_code");
    Exit_USER (user_exit_msg);
    free (parameter_dimension);
    free (USER_OPTIONS);
    return (-2);
  }
  if ((cost_flag = (int *) calloc (1, sizeof (int))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): cost_flag");
    Exit_USER (user_exit_msg);
    free (parameter_dimension);
    free (exit_code);
    free (USER_OPTIONS);
    return (-2);
  }
#if OPTIONS_FILE
  /* Test to see if asa_opt is in correct directory.
     This is useful for some PC and Mac compilers. */
  if ((ptr_options = fopen ("asa_opt", "r")) == NULL) {
    fprintf (ptr_out, "\n\n*** fopen asa_opt failed *** \n\n");
    fflush (ptr_out);
#if INCL_STDOUT
    printf ("\n\n*** EXIT fopen asa_opt failed *** \n\n");
#endif /* INCL_STDOUT */
    free (parameter_dimension);
    free (exit_code);
    free (cost_flag);
    free (USER_OPTIONS);
    return (-6);
  }

  fscanf_ret = fscanf (ptr_options, "%s%s%s%s%s",
                       read_if, read_FALSE, read_comm1, read_ASA_SAVE,
                       read_comm2);
  if (strcmp (read_if, "#if") || strcmp (read_FALSE, "FALSE")
      || strcmp (read_comm1, "/*") || strcmp (read_ASA_SAVE, "ASA_SAVE")
      || strcmp (read_comm2, "*/")) {
    fprintf (ptr_out, "\n\n*** not asa_opt for this version *** \n\n");
    fflush (ptr_out);
#if INCL_STDOUT
    printf ("\n\n*** EXIT not asa_opt for this version *** \n\n");
#endif /* INCL_STDOUT */
    free (parameter_dimension);
    free (exit_code);
    free (cost_flag);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-6);
  }
#if INT_LONG
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%ld", &read_long);
  USER_OPTIONS->Limit_Acceptances = read_long;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%ld", &read_long);
  USER_OPTIONS->Limit_Generated = read_long;
#else
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Limit_Acceptances = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Limit_Generated = read_int;
#endif
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Limit_Invalid_Generated_States = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Accepted_To_Generated_Ratio = read_double;

  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Cost_Precision = read_double;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Maximum_Cost_Repeat = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Number_Cost_Samples = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Temperature_Ratio_Scale = read_double;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Cost_Parameter_Scale_Ratio = read_double;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Temperature_Anneal_Scale = read_double;

  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Include_Integer_Parameters = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->User_Initial_Parameters = read_int;
#if INT_ALLOC
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Sequential_Parameters = read_int;
#else
#if INT_LONG
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%ld", &read_long);
  USER_OPTIONS->Sequential_Parameters = read_long;
#else
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Sequential_Parameters = read_int;
#endif
#endif
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Initial_Parameter_Temperature = read_double;

  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Acceptance_Frequency_Modulus = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Generated_Frequency_Modulus = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Reanneal_Cost = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Reanneal_Parameters = read_int;

  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Delta_X = read_double;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->User_Tangents = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Curvature_0 = read_int;

#else /* OPTIONS_FILE */
  /* USER_OPTIONS->Limit_Acceptances = 10000; */
  USER_OPTIONS->Limit_Acceptances = 1000;
  USER_OPTIONS->Limit_Generated = 99999;
  USER_OPTIONS->Limit_Invalid_Generated_States = 1000;
  /* USER_OPTIONS->Accepted_To_Generated_Ratio = 1.0E-6; */
  USER_OPTIONS->Accepted_To_Generated_Ratio = 1.0E-4;

  USER_OPTIONS->Cost_Precision = 1.0E-18;
  USER_OPTIONS->Maximum_Cost_Repeat = 5;
  USER_OPTIONS->Number_Cost_Samples = 5;
  USER_OPTIONS->Temperature_Ratio_Scale = 1.0E-5;
  USER_OPTIONS->Cost_Parameter_Scale_Ratio = 1.0;
  USER_OPTIONS->Temperature_Anneal_Scale = 100.0;

  USER_OPTIONS->Include_Integer_Parameters = FALSE;
  USER_OPTIONS->User_Initial_Parameters = FALSE;
  USER_OPTIONS->Sequential_Parameters = -1;
  USER_OPTIONS->Initial_Parameter_Temperature = 1.0;

  USER_OPTIONS->Acceptance_Frequency_Modulus = 100;
  USER_OPTIONS->Generated_Frequency_Modulus = 10000;
  USER_OPTIONS->Reanneal_Cost = 1;
  USER_OPTIONS->Reanneal_Parameters = TRUE;

  USER_OPTIONS->Delta_X = 0.001;
  USER_OPTIONS->User_Tangents = FALSE;
  USER_OPTIONS->Curvature_0 = FALSE;

#endif /* OPTIONS_FILE */

  USER_OPTIONS->Limit_Generated = TargetIter;
  /* ALLOCATE STORAGE */

#if ASA_SAVE
  /* Such data could be saved in a user_save file, but for
     convenience here everything is saved in asa_save. */
  USER_OPTIONS->Random_Array_Dim = SHUFFLE;
  USER_OPTIONS->Random_Array = random_array;
#endif /* ASA_SAVE */

#if USER_ASA_OUT
  if ((USER_OPTIONS->Asa_Out_File =
       (char *) calloc (80, sizeof (char))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): USER_OPTIONS->Asa_Out_File");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  sprintf (USER_OPTIONS->Asa_Out_File, "%s", "asa_out_my");

  /* OR create memory for file asa_out_my */
  /* which must be "free(asa_out_my);" after use */
  strcpy (USER_OPTIONS->Asa_Out_File, asa_out_my);

  ptr_out = fopen (USER_OPTIONS->Asa_Out_File, "w");
#endif /* ASA_TEMPLATE */
#endif /* USER_ASA_OUT */

  /* the number of parameters for the cost function */
#if OPTIONS_FILE_DATA
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%s", read_option);

#if INT_ALLOC
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  *parameter_dimension = read_int;
#else
#if INT_LONG
  fscanf_ret = fscanf (ptr_options, "%ld", &read_long);
  *parameter_dimension = read_long;
#else
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  *parameter_dimension = read_int;
#endif
#endif

#else /* OPTIONS_FILE_DATA */
#if ASA_TEST
  *parameter_dimension = 4;
#endif /* ASA_TEST */
#endif /* OPTIONS_FILE_DATA */
#if MY_TEMPLATE                 /* MY_TEMPLATE_dim */
  /* If not using OPTIONS_FILE_DATA or data read from asa_opt,
     insert the number of parameters for the cost_function */
#endif /* MY_TEMPLATE dim */

#if ASA_TEMPLATE_SAMPLE
  *parameter_dimension = 2;
  USER_OPTIONS->Limit_Acceptances = 2000;
  USER_OPTIONS->User_Tangents = TRUE;
  USER_OPTIONS->Limit_Weights = 1.0E-7;
#endif
#if ASA_TEMPLATE_PARALLEL
  USER_OPTIONS->Gener_Block = 100;
  USER_OPTIONS->Gener_Block_Max = 512;
  USER_OPTIONS->Gener_Mov_Avr = 3;
#endif

  /* allocate parameter minimum space */
  if ((parameter_lower_bound =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
	  printf("%d\n",*parameter_dimension);
    strcpy (user_exit_msg, "main()/asa_main(): parameter_lower_bound");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (cost_flag);
    free (parameter_dimension);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  /* allocate parameter maximum space */
  if ((parameter_upper_bound =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): parameter_upper_bound");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_lower_bound);
    free (parameter_dimension);
    free (cost_flag);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  /* allocate parameter initial values; the parameter final values
     will be stored here later */
  if ((cost_parameters =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): cost_parameters");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (parameter_dimension);
    free (cost_parameters);
    free (cost_flag);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  /* allocate the parameter types, real or integer */
  if ((parameter_int_real =
       (int *) calloc (*parameter_dimension, sizeof (int))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): parameter_int_real");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (parameter_dimension);
    free (cost_parameters);
    free (cost_flag);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  /* allocate space for parameter cost_tangents -
     used for reannealing */
  if ((cost_tangents =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): cost_tangents");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (parameter_dimension);
    free (cost_parameters);
    free (cost_tangents);
    free (parameter_int_real);
    free (cost_flag);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }

  if ((USER_OPTIONS->Curvature_0 == FALSE)
      || (USER_OPTIONS->Curvature_0 == -1)) {
    /* allocate space for parameter cost_curvatures/covariance */
    if ((cost_curvature =
         (double *) calloc ((*parameter_dimension) *
                            (*parameter_dimension),
                            sizeof (double))) == NULL) {
      strcpy (user_exit_msg, "main()/asa_main(): cost_curvature");
      Exit_USER (user_exit_msg);
      free (exit_code);
      free (parameter_lower_bound);
      free (parameter_upper_bound);
      free (cost_parameters);
      free (cost_tangents);
      free (parameter_dimension);
      free (parameter_int_real);
      free (cost_flag);
      fclose (ptr_options);
      return (-2);
    }
  } else {
    cost_curvature = (double *) NULL;
  }

#if USER_COST_SCHEDULE
  USER_OPTIONS->Cost_Schedule = user_cost_schedule;
#endif
#if USER_ACCEPTANCE_TEST
  USER_OPTIONS->Acceptance_Test = user_acceptance_test;
#endif
#if USER_ACCEPT_ASYMP_EXP
  USER_OPTIONS->Asymp_Exp_Param = 1.0;
#endif
#if USER_GENERATING_FUNCTION
  USER_OPTIONS->Generating_Distrib = user_generating_distrib;
#endif
#if USER_REANNEAL_COST
  USER_OPTIONS->Reanneal_Cost_Function = user_reanneal_cost;
#endif
#if USER_REANNEAL_PARAMETERS
  USER_OPTIONS->Reanneal_Params_Function = user_reanneal_params;
#endif

#if MY_TEMPLATE                 /* MY_TEMPLATE_pre_initialize */
  /* last changes before entering initialize_parameters() */
#endif

  initialize_parameters_value = initialize_parameters (cost_parameters,
                                                       parameter_lower_bound,
                                                       parameter_upper_bound,
                                                       cost_tangents,
                                                       cost_curvature,
                                                       parameter_dimension,
                                                       parameter_int_real,
#if OPTIONS_FILE_DATA
                                                       ptr_options,
#endif
                                                       USER_OPTIONS);
#if OPTIONS_FILE
  fclose (ptr_options);
#endif
  if (initialize_parameters_value == -2) {
    free (exit_code);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (cost_parameters);
    free (cost_tangents);
    free (parameter_dimension);
    free (parameter_int_real);
    free (cost_flag);
    free (USER_OPTIONS);
    free (cost_curvature);
    return (initialize_parameters_value);
  }

  /* optimize the cost_function, returning the results in
     cost_value and cost_parameters */
#if ASA_TEMPLATE_MULTIPLE
  /* multiple asa() quenched calls + multiple asa_out files
     (To get longer quenched runs, decrease SMALL_FLOAT.) */
  for (n_asa = 1; n_asa <= *parameter_dimension; n_asa++) {
    asa_file[4] = 'A' + n_asa - 1;
    USER_OPTIONS->User_Quench_Cost_Scale[0] = (double) n_asa;
    for (index = 0; index < *parameter_dimension; ++index)
      USER_OPTIONS->User_Quench_Param_Scale[index] = (double) n_asa;
    for (n_trajectory = 0; n_trajectory < 3; ++n_trajectory) {
      asa_file[6] = 'a' + n_trajectory;
      strcpy (USER_OPTIONS->Asa_Out_File, asa_file);
#endif

#if ASA_TEMPLATE_ASA_OUT_PID
      pid_file[0] = 'a';
      pid_file[1] = 's';
      pid_file[2] = 'a';
      pid_file[3] = '_';
      pid_file[4] = 'o';
      pid_file[5] = 'u';
      pid_file[6] = 't';
      pid_file[7] = '_';

      pid_int = getpid ();
      if (pid_int < 0) {
        pid_file[7] = '0';
        pid_int = -pid_int;
      }

      strcpy (USER_OPTIONS->Asa_Out_File, pid_file);
#endif
#if ASA_FUZZY
      InitFuzzyASA (USER_OPTIONS, *parameter_dimension);
#endif /* ASA_FUZZY */
      cost_value =
        asa (USER_COST_FUNCTION,
             randflt,
             rand_seed,
             cost_parameters,
             parameter_lower_bound,
             parameter_upper_bound,
             cost_tangents,
             cost_curvature,
             parameter_dimension,
             parameter_int_real, cost_flag, exit_code, USER_OPTIONS);
      if (*exit_code == -1) {
#if INCL_STDOUT
        printf ("\n\n*** error in calloc in ASA ***\n\n");
#endif /* INCL_STDOUT */
        fprintf (ptr_out, "\n\n*** error in calloc in ASA ***\n\n");
        fflush (ptr_out);
        return (-1);
      }
#if ASA_FUZZY
      if (USER_OPTIONS->Locate_Cost == 12) {
        USER_OPTIONS->Locate_Cost = 0;
      }
      CloseFuzzyASA (USER_OPTIONS);
#endif /* ASA_FUZZY */

#if MULTI_MIN
      fprintf (ptr_out, "Multi_Specify = %d\n", USER_OPTIONS->Multi_Specify);
#if INT_LONG
      fprintf (ptr_out, "N_Accepted = %ld\n", USER_OPTIONS->N_Accepted);
#else
      fprintf (ptr_out, "N_Accepted = %d\n", USER_OPTIONS->N_Accepted);
#endif
#if ASA_RESOLUTION
      for (n_param = 0; n_param < *parameter_dimension; ++n_param) {
        fprintf (ptr_out,
#if INT_ALLOC
                 "Coarse_Resolution[%d] = %12.7g\n",
#else
#if INT_LONG
                 "Coarse_Resolution[%ld] = %12.7g\n",
#else
                 "Coarse_Resolution[%d] = %12.7g\n",
#endif
#endif
                 n_param, USER_OPTIONS->Coarse_Resolution[n_param]);
      }
#else /* ASA_RESOLUTION */
      for (n_param = 0; n_param < *parameter_dimension; ++n_param) {
        fprintf (ptr_out,
#if INT_ALLOC
                 "Multi_Grid[%d] = %12.7g\n",
#else
#if INT_LONG
                 "Multi_Grid[%ld] = %12.7g\n",
#else
                 "Multi_Grid[%d] = %12.7g\n",
#endif
#endif
                 n_param, USER_OPTIONS->Multi_Grid[n_param]);
      }
#endif /* ASA_RESOLUTION */
      fprintf (ptr_out, "\n");
      for (multi_index = 0; multi_index < USER_OPTIONS->Multi_Number;
           ++multi_index) {
        fprintf (ptr_out, "\n");
        fprintf (ptr_out, "Multi_Cost[%d] = %12.7g\n",
                 multi_index, USER_OPTIONS->Multi_Cost[multi_index]);
        for (n_param = 0; n_param < *parameter_dimension; ++n_param) {
          fprintf (ptr_out,
#if INT_ALLOC
                   "Multi_Params[%d][%d] = %12.7g\n",
#else
#if INT_LONG
                   "Multi_Params[%d][%ld] = %12.7g\n",
#else
                   "Multi_Params[%d][%d] = %12.7g\n",
#endif
#endif
                   multi_index, n_param,
                   USER_OPTIONS->Multi_Params[multi_index][n_param]);
        }
      }
      fprintf (ptr_out, "\n");
      fflush (ptr_out);

      cost_value = USER_OPTIONS->Multi_Cost[0];
      for (n_param = 0; n_param < *parameter_dimension; ++n_param) {
        cost_parameters[n_param] = USER_OPTIONS->Multi_Params[0][n_param];
      }
#endif /* MULTI_MIN */

#if FITLOC
      /* Fit_Local, Iter_Max and Penalty may be set adaptively */
      USER_OPTIONS->Penalty = 1000;
      USER_OPTIONS->Fit_Local = 1;
      USER_OPTIONS->Iter_Max = 500;
      if (USER_OPTIONS->Fit_Local >= 1) {
        cost_value = fitloc (USER_COST_FUNCTION,
                             cost_parameters,
                             parameter_lower_bound,
                             parameter_upper_bound,
                             cost_tangents,
                             cost_curvature,
                             parameter_dimension,
                             parameter_int_real,
                             cost_flag, exit_code, USER_OPTIONS, ptr_out);
      }
#endif /* FITLOC */

#if ASA_TEMPLATE                /* extra USER_COST_FUNCTION run */
      /* If your USER_COST_FUNCTION modifies your other programs, and final
       * calls to asa() and/or fitloc() make additional modifications,
       * you might run a last call to USER_COST_FUNCTION(). */
      cost_value = USER_COST_FUNCTION (cost_parameters,
                                       parameter_lower_bound,
                                       parameter_upper_bound,
                                       cost_tangents,
                                       cost_curvature,
                                       parameter_dimension,
                                       parameter_int_real,
                                       cost_flag, exit_code, USER_OPTIONS);
#endif /* extra USER_COST_FUNCTION run */

#if MY_TEMPLATE                 /* MY_TEMPLATE_post_asa */
#endif
#if ASA_TEMPLATE_LIB
      *main_cost_value = cost_value;
      for (n_param = 0; n_param < *parameter_dimension; ++n_param) {
        main_cost_parameters[n_param] = cost_parameters[n_param];
      }
      *main_exit_code = *exit_code;
#endif

      fprintf (ptr_out, "exit code = %d\n", *exit_code);
      fprintf (ptr_out, "final cost value = %12.7g\n", cost_value);
      fprintf (ptr_out, "parameter\tvalue\n");
      for (n_param = 0; n_param < *parameter_dimension; ++n_param) {
        fprintf (ptr_out,
#if INT_ALLOC
                 "%d\t\t%12.7g\n",
#else
#if INT_LONG
                 "%ld\t\t%12.7g\n",
#else
                 "%d\t\t%12.7g\n",
#endif
#endif
                 n_param, cost_parameters[n_param]);
      }

#if TIME_CALC
      /* print ending time */
      print_time ("end", ptr_out);
#endif
#if ASA_TEMPLATE_MULTIPLE
    }
  }
#endif

#if ASA_TEMPLATE_SAMPLE
  ptr_asa = fopen ("asa_out", "r");
  sample (ptr_out, ptr_asa);
#endif

  /* close all files */
  fclose (ptr_out);
#if OPTIONAL_DATA_DBL
  free (USER_OPTIONS->Asa_Data_Dbl);
#endif
#if OPTIONAL_DATA_INT
  free (USER_OPTIONS->Asa_Data_Int);
#endif
#if OPTIONAL_DATA_PTR
#if MY_TEMPLATE
  /* Instead of freeing Asa_Data_Ptr, if memory has been allocated
   * outside ASA, e.g., by the use of ASA_LIB, use the following: */
  /* USER_OPTIONS->Asa_Data_Ptr = NULL; */
#endif /* MY_TEMPLATE */
  free (USER_OPTIONS->Asa_Data_Ptr);
#endif
#if USER_ASA_OUT
#if TEMPLATE
  /* if necessary */
  free (asa_out_my);
#endif
  free (USER_OPTIONS->Asa_Out_File);
#endif
#if USER_ASA_USR_OUT
#if ASA_TEMPLATE
  /* if necessary */
  free (asa_usr_out_my);
#endif
  free (USER_OPTIONS->Asa_Usr_Out_File);
#endif
#if ASA_SAMPLE
  free (USER_OPTIONS->Bias_Generated);
#endif
#if ASA_QUEUE
#if ASA_RESOLUTION
#else
  free (USER_OPTIONS->Queue_Resolution);
#endif
#endif
#if ASA_RESOLUTION
  free (USER_OPTIONS->Coarse_Resolution);
#endif
#if USER_INITIAL_PARAMETERS_TEMPS
  free (USER_OPTIONS->User_Parameter_Temperature);
#endif
#if USER_INITIAL_COST_TEMP
  free (USER_OPTIONS->User_Cost_Temperature);
#endif
#if DELTA_PARAMETERS
  free (USER_OPTIONS->User_Delta_Parameter);
#endif
#if QUENCH_PARAMETERS
  free (USER_OPTIONS->User_Quench_Param_Scale);
#endif
#if QUENCH_COST
  free (USER_OPTIONS->User_Quench_Cost_Scale);
#endif
#if RATIO_TEMPERATURE_SCALES
  free (USER_OPTIONS->User_Temperature_Ratio);
#endif
#if MULTI_MIN
  free (USER_OPTIONS->Multi_Cost);
  free (USER_OPTIONS->Multi_Grid);
  for (multi_index = 0; multi_index < USER_OPTIONS->Multi_Number;
       ++multi_index) {
    free (USER_OPTIONS->Multi_Params[multi_index]);
  }
  free (USER_OPTIONS->Multi_Params);
#endif /* MULTI_MIN */
  free (USER_OPTIONS);
  free (parameter_dimension);
  free (exit_code);
  free (cost_flag);
  free (parameter_lower_bound);
  free (parameter_upper_bound);
  free (cost_parameters);
  free (parameter_int_real);
  free (cost_tangents);
  free (rand_seed);
  free (cost_curvature);

  end_time = PAPI_get_real_usec();
  //printf("Time = %lld us \n", end_time - start_time); 

  return (0);
}
#endif /* SELF_OPTIMIZE */

/***********************************************************************
* initialize_parameters - sample parameter initialization function
*	This depends on the users cost function to optimize (minimum).
*	The routine allocates storage needed for asa. The user should
*	define the number of parameters and their ranges,
*	and make sure the initial parameters are within
*	the minimum and maximum ranges. The array
*	parameter_int_real should be REAL_TYPE (-1) for real parameters,
*	and INTEGER_TYPE (1) for integer values
***********************************************************************/
#if HAVE_ANSI
int
initialize_parameters (double *cost_parameters,
                       double *parameter_lower_bound,
                       double *parameter_upper_bound,
                       double *cost_tangents,
                       double *cost_curvature,
                       ALLOC_INT * parameter_dimension,
                       int *parameter_int_real,
#if OPTIONS_FILE_DATA
                       FILE * ptr_options,
#endif
                       USER_DEFINES * USER_OPTIONS)
#else
int
initialize_parameters (cost_parameters,
                       parameter_lower_bound,
                       parameter_upper_bound,
                       cost_tangents,
                       cost_curvature,
                       parameter_dimension, parameter_int_real,
#if OPTIONS_FILE_DATA
                       ptr_options,
#endif
                       USER_OPTIONS)
     double *cost_parameters;
     double *parameter_lower_bound;
     double *parameter_upper_bound;
     double *cost_tangents;
     double *cost_curvature;
     ALLOC_INT *parameter_dimension;
     int *parameter_int_real;
#if OPTIONS_FILE_DATA
     FILE *ptr_options;
#endif
     USER_DEFINES *USER_OPTIONS;
#endif
{
  ALLOC_INT index;
#if OPTIONS_FILE_DATA
  int fscanf_ret;
  char read_option[80];
  ALLOC_INT read_index;
#endif
#if MULTI_MIN
  int multi_index;
#endif
#if MY_TEMPLATE                 /* MY_TEMPLATE_init_decl */
  /* add some declarations if required */
#endif

  index = 0;
#if OPTIONS_FILE_DATA
  fscanf_ret = 0;               /* stop compiler warning */
  if (fscanf_ret) {
    ;
  }

  fscanf_ret = fscanf (ptr_options, "%s", read_option);

  for (index = 0; index < *parameter_dimension; ++index) {
#if MY_TEMPLATE                 /* MY_TEMPLATE_read_opt */
    /* put in some code as required to alter lines read from asa_opt */
#endif
#if INT_ALLOC
    fscanf_ret = fscanf (ptr_options, "%d", &read_index);
#else
#if INT_LONG
    fscanf_ret = fscanf (ptr_options, "%ld", &read_index);
#else
    fscanf_ret = fscanf (ptr_options, "%d", &read_index);
#endif
#endif
    fscanf_ret = fscanf (ptr_options, "%lf%lf%lf%d",
                         &(parameter_lower_bound[read_index]),
                         &(parameter_upper_bound[read_index]),
                         &(cost_parameters[read_index]),
                         &(parameter_int_real[read_index]));
  }
#else /* OPTIONS_FILE_DATA */
#if ASA_TEST
  /* store the parameter ranges */
  for (index = 0; index < *parameter_dimension; ++index)
    parameter_lower_bound[index] = -10000.0;
  for (index = 0; index < *parameter_dimension; ++index)
    parameter_upper_bound[index] = 10000.0;

  /* store the initial parameter types */
  for (index = 0; index < *parameter_dimension; ++index)
    parameter_int_real[index] = REAL_TYPE;

  /* store the initial parameter values */
  for (index = 0; index < *parameter_dimension / 4.0; ++index) {
    cost_parameters[4 * (index + 1) - 4] = 999.0;
    cost_parameters[4 * (index + 1) - 3] = -1007.0;
    cost_parameters[4 * (index + 1) - 2] = 1001.0;
    cost_parameters[4 * (index + 1) - 1] = -903.0;
  }
#endif /* ASA_TEST */
#endif /* OPTIONS_FILE_DATA */
#if ASA_TEMPLATE_SAMPLE
  for (index = 0; index < *parameter_dimension; ++index)
    parameter_lower_bound[index] = 0;
  for (index = 0; index < *parameter_dimension; ++index)
    parameter_upper_bound[index] = 2.0;
  for (index = 0; index < *parameter_dimension; ++index)
    parameter_int_real[index] = REAL_TYPE;
  for (index = 0; index < *parameter_dimension; ++index)
    cost_parameters[index] = 0.5;
#endif

#if USER_INITIAL_PARAMETERS_TEMPS
  if ((USER_OPTIONS->User_Parameter_Temperature =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->User_Parameter_Temperature");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  for (index = 0; index < *parameter_dimension; ++index)
    USER_OPTIONS->User_Parameter_Temperature[index] = 1.0;
#endif
#endif /* USER_INITIAL_PARAMETERS_TEMPS */
#if USER_INITIAL_COST_TEMP
  if ((USER_OPTIONS->User_Cost_Temperature =
       (double *) calloc (1, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->User_Cost_Temperature");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  USER_OPTIONS->User_Cost_Temperature[0] = 5.936648E+09;
#endif
#endif /* USER_INITIAL_COST_TEMP */
#if DELTA_PARAMETERS
  if ((USER_OPTIONS->User_Delta_Parameter =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->User_Delta_Parameter");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  for (index = 0; index < *parameter_dimension; ++index)
    USER_OPTIONS->User_Delta_Parameter[index] = 0.001;
#endif
#endif /* DELTA_PARAMETERS */
#if QUENCH_PARAMETERS
  if ((USER_OPTIONS->User_Quench_Param_Scale =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->User_Quench_Param_Scale");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  for (index = 0; index < *parameter_dimension; ++index)
    USER_OPTIONS->User_Quench_Param_Scale[index] = 1.0;
#endif
#if ASA_TEMPLATE_MULTIPLE
  for (index = 0; index < *parameter_dimension; ++index)
    USER_OPTIONS->User_Quench_Param_Scale[index] = 1.0;
#endif
#if ASA_TEMPLATE_SAVE
  for (index = 0; index < *parameter_dimension; ++index)
    USER_OPTIONS->User_Quench_Param_Scale[index] = 1.0;
#endif
#endif /* QUENCH_PARAMETERS */
#if QUENCH_COST
  if ((USER_OPTIONS->User_Quench_Cost_Scale =
       (double *) calloc (1, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->User_Quench_Cost_Scale");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  USER_OPTIONS->User_Quench_Cost_Scale[0] = 1.0;
#endif
#if ASA_TEMPLATE_MULTIPLE
  USER_OPTIONS->User_Quench_Cost_Scale[0] = 1.0;
#endif
#if ASA_TEMPLATE_SAVE
  USER_OPTIONS->User_Quench_Cost_Scale[0] = 1.0;
#endif
#endif /* QUENCH_COST */

  /* use asa_opt to read in QUENCH USER_OPTIONS */
#if OPTIONS_FILE_DATA
#if QUENCH_COST
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret =
    fscanf (ptr_options, "%lf", &(USER_OPTIONS->User_Quench_Cost_Scale[0]));

#if QUENCH_PARAMETERS
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  for (index = 0; index < *parameter_dimension; ++index) {
#if INT_ALLOC
    fscanf_ret = fscanf (ptr_options, "%d", &read_index);
#else
#if INT_LONG
    fscanf_ret = fscanf (ptr_options, "%ld", &read_index);
#else
    fscanf_ret = fscanf (ptr_options, "%d", &read_index);
#endif
#endif
    fscanf_ret = fscanf (ptr_options, "%lf",
                         &(USER_OPTIONS->User_Quench_Param_Scale
                           [read_index]));
  }
#endif /* QUENCH_PARAMETERS */
#endif /* QUENCH_COST */
#endif /* OPTIONS_FILE_DATA */

#if RATIO_TEMPERATURE_SCALES
  if ((USER_OPTIONS->User_Temperature_Ratio =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->User_Temperature_Ratio");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  for (index = 0; index < *parameter_dimension; ++index)
    USER_OPTIONS->User_Temperature_Ratio[index] = 1.0;
#endif
#endif /* RATIO_TEMPERATURE_SCALES */
  /* Defines the limit of collection of sampled data by asa */
#if ASA_SAMPLE
  /* create memory for Bias_Generated[] */
  if ((USER_OPTIONS->Bias_Generated =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->Bias_Generated");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#endif

#if ASA_RESOLUTION
  if ((USER_OPTIONS->Coarse_Resolution =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->Coarse_Resolution");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  for (index = 0; index < *parameter_dimension; ++index)
    USER_OPTIONS->Coarse_Resolution[index] = 1.0;
#endif
#endif /* ASA_RESOLUTION */
#if ASA_QUEUE
#if ASA_RESOLUTION
  USER_OPTIONS->Queue_Resolution = USER_OPTIONS->Coarse_Resolution;
#else /* ASA_RESOLUTION */
  if ((USER_OPTIONS->Queue_Resolution =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->Queue_Resolution");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE_QUEUE
  for (index = 0; index < *parameter_dimension; ++index)
    USER_OPTIONS->Queue_Resolution[index] = 0.001;
#endif
#endif /* ASA_RESOLUTION */
#if ASA_TEMPLATE_QUEUE
  USER_OPTIONS->Queue_Size = 100;
#endif
#endif /* ASA_QUEUE */
#if MULTI_MIN
#if ASA_TEMPLATE
  USER_OPTIONS->Multi_Number = 2;
#endif
  if ((USER_OPTIONS->Multi_Cost =
       (double *) calloc (USER_OPTIONS->Multi_Number,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->Multi_Cost");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  if ((USER_OPTIONS->Multi_Grid =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->Multi_Grid");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  if ((USER_OPTIONS->Multi_Params =
       (double **) calloc (USER_OPTIONS->Multi_Number,
                           sizeof (double *))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): USER_OPTIONS->Multi_Params");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  for (multi_index = 0; multi_index < USER_OPTIONS->Multi_Number;
       ++multi_index) {
    if ((USER_OPTIONS->Multi_Params[multi_index] =
         (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
      strcpy (user_exit_msg,
              "initialize_parameters(): USER_OPTIONS->Multi_Params[multi_index]");
      Exit_USER (user_exit_msg);
      return (-2);
    }
  }
#if ASA_TEST
  for (index = 0; index < *parameter_dimension; ++index) {
    USER_OPTIONS->Multi_Grid[index] = 0.05;
  }
  USER_OPTIONS->Multi_Specify = 0;
#endif
#if ASA_TEMPLATE
  for (index = 0; index < *parameter_dimension; ++index) {
    USER_OPTIONS->Multi_Grid[index] =
      (parameter_upper_bound[index] - parameter_lower_bound[index]) / 100.0;
  }
  USER_OPTIONS->Multi_Specify = 0;
#endif /* ASA_TEMPLATE */
#endif /* MULTI_MIN */
  USER_OPTIONS->Asa_Recursive_Level = 0;

#if MY_TEMPLATE                 /* MY_TEMPLATE_params */
  /* If not using RECUR_OPTIONS_FILE_DATA or data read from asa_opt,
     store the parameter ranges
     store the parameter types
     store the initial parameter values
     other changes needed for initialization */
#endif /* MY_TEMPLATE params */

  return (0);
}

#if COST_FILE
#else
/***********************************************************************
* double cost_function
*	This is the users cost function to optimize
*	(find the minimum).
*	cost_flag is set to TRUE if the parameter set
*	does not violates any constraints
*       parameter_lower_bound and parameter_upper_bound may be
*       adaptively changed during the search.
***********************************************************************/

#if HAVE_ANSI
double
cost_function (double *x,
               double *parameter_lower_bound,
               double *parameter_upper_bound,
               double *cost_tangents,
               double *cost_curvature,
               ALLOC_INT * parameter_dimension,
               int *parameter_int_real,
               int *cost_flag, int *exit_code, USER_DEFINES * USER_OPTIONS)
#else
double
cost_function (x,
               parameter_lower_bound,
               parameter_upper_bound,
               cost_tangents,
               cost_curvature,
               parameter_dimension,
               parameter_int_real, cost_flag, exit_code, USER_OPTIONS)
     double *x;
     double *parameter_lower_bound;
     double *parameter_upper_bound;
     double *cost_tangents;
     double *cost_curvature;
     ALLOC_INT *parameter_dimension;
     int *parameter_int_real;
     int *cost_flag;
     int *exit_code;
     USER_DEFINES *USER_OPTIONS;
#endif
{

#if ASA_TEST                    /* ASA test problem */
  /* Objective function from
   * %A A. Corana
   * %A M. Marchesi
   * %A C. Martini
   * %A S. Ridella
   * %T Minimizing multimodal functions of continuous variables
   *    with the "simulated annealing" algorithm
   * %J ACM Trans. Mathl. Software
   * %V 13
   * %N 3
   * %P 262-279
   * %D 1987
   *
   * This function, when used with ASA_TEST_POINT set to TRUE, contains
   * 1.0E20 local minima.  When *parameter_dimension is equal to 4, visiting
   * each minimum for a millisecond would take about the present age of the
   * universe to visit all these minima. */

  /* defines for the test problem, which assume *parameter_dimension
     is a multiple of 4.  If this is set to a large number, you
     likely should set Curvature_0 to TRUE. */
  double q_n, d_i, s_i, t_i, z_i, c_r;
  int k_i;
#if ASA_TEST_POINT
  ALLOC_INT k_flag;
#endif
  ALLOC_INT i, j;
#if SELF_OPTIMIZE
#else
  static LONG_INT funevals = 0;
#endif
#if ASA_TEMPLATE_SAVE
  static int read_test = 0;
  FILE *ptr_read_test;
#endif

#if ADAPTIVE_OPTIONS
  adaptive_options (USER_OPTIONS);
#endif

#if MY_TEMPLATE                 /* MY_TEMPLATE_diminishing_ranges */
  /* insert code to automate changing ranges of parameters */
#endif
#if ASA_TEMPLATE                /* example of diminishing ranges */
  if (USER_OPTIONS->Locate_Cost == 12 && *(USER_OPTIONS->Best_Cost) < 1.0) {
    fprintf (ptr_out, "best_cost = %g\n", *(USER_OPTIONS->Best_Cost));
    for (i = 0; i < *parameter_dimension; ++i) {
      parameter_lower_bound[i] = USER_OPTIONS->Best_Parameters[i]
        - 0.5 * fabs (parameter_lower_bound[i]
                      - USER_OPTIONS->Best_Parameters[i]);
      parameter_upper_bound[i] = USER_OPTIONS->Best_Parameters[i]
        + 0.5 * fabs (parameter_upper_bound[i]
                      - USER_OPTIONS->Best_Parameters[i]);
      parameter_lower_bound[i] = MIN (parameter_lower_bound[i],
                                      USER_OPTIONS->Best_Parameters[i] -
                                      0.01);
      parameter_upper_bound[i] =
        MAX (parameter_upper_bound[i],
             USER_OPTIONS->Best_Parameters[i] + 0.01);
    }
  }
#endif /* ASA_TEMPLATE */

  /* a_i = parameter_upper_bound[i] */
  s_i = 0.2;
  t_i = 0.05;
  c_r = 0.15;

#if ASA_TEST_POINT
  k_flag = 0;
  for (i = 0; i < *parameter_dimension; ++i) {
    if (x[i] > 0.0) {
      k_i = (int) (x[i] / s_i + 0.5);
    } else if (x[i] < 0.0) {
      k_i = (int) (x[i] / s_i - 0.5);
    } else {
      k_i = 0;
    }
    if (k_i == 0)
      ++k_flag;
  }
#endif /* ASA_TEST_POINT */

  q_n = 0.0;
  for (i = 0; i < *parameter_dimension; ++i) {
    j = i % 4;
    switch (j) {
    case 0:
      d_i = 1.0;
      break;
    case 1:
      d_i = 1000.0;
      break;
    case 2:
      d_i = 10.0;
      break;
    default:
      d_i = 100.0;
    }
    if (x[i] > 0.0) {
      k_i = (int) (x[i] / s_i + 0.5);
    } else if (x[i] < 0.0) {
      k_i = (int) (x[i] / s_i - 0.5);
    } else {
      k_i = 0;
    }

#if ASA_TEST_POINT
    if (fabs (k_i * s_i - x[i]) < t_i && k_flag != *parameter_dimension)
#else
    if (fabs (k_i * s_i - x[i]) < t_i)
#endif
    {
      if (k_i < 0) {
        z_i = k_i * s_i + t_i;
      } else if (k_i > 0) {
        z_i = k_i * s_i - t_i;
      } else {
        z_i = 0.0;
      }
      q_n += c_r * d_i * z_i * z_i;
    } else {
      q_n += d_i * x[i] * x[i];
    }
  }
  funevals = funevals + 1;

#if ASA_TEMPLATE_SAVE
  /* cause a crash */
  if ((ptr_read_test = fopen ("asa_save", "r")) == NULL) {
    read_test = 1;
  } else {
    fclose (ptr_read_test);
  }
  /* will need a few hundred if testing ASA_PARALLEL to get an asa_save */
  if (funevals == 50 && read_test == 1) {
    fprintf (ptr_out, "\n\n*** intended crash to test ASA_SAVE *** \n\n");
    fflush (ptr_out);
#if INCL_STDOUT
    printf ("\n\n*** intended crash to test ASA_SAVE *** \n\n");
#endif /* INCL_STDOUT */
    exit (2);
  }
#endif

  *cost_flag = TRUE;

#if SELF_OPTIMIZE
#else
#if TIME_CALC
  /* print the time every PRINT_FREQUENCY evaluations */
  if ((PRINT_FREQUENCY > 0) && ((funevals % PRINT_FREQUENCY) == 0)) {
    fprintf (ptr_out, "funevals = %ld  ", funevals);
#if INCL_STDOUT
    print_time ("", ptr_out);
#endif /* INCL_STDOUT */
  }
#endif
#endif

#if ASA_TEMPLATE_SAMPLE
  USER_OPTIONS->Cost_Acceptance_Flag = TRUE;
  if (USER_OPTIONS->User_Acceptance_Flag == FALSE && *cost_flag == TRUE)
    USER_OPTIONS->Acceptance_Test (q_n,
                                   parameter_lower_bound,
                                   parameter_upper_bound,
                                   *parameter_dimension, USER_OPTIONS);
#endif /* ASA_TEMPLATE_SAMPLE */

#if ASA_FUZZY
  if (*cost_flag == TRUE
      && (USER_OPTIONS->Locate_Cost == 2 || USER_OPTIONS->Locate_Cost == 3
          || USER_OPTIONS->Locate_Cost == 4)) {
    FuzzyControl (USER_OPTIONS, x, q_n, *parameter_dimension);
  }
#endif /* ASA_FUZZY */

  return (q_n);
#endif /* ASA_TEST */
#if ASA_TEMPLATE_SAMPLE

  int n;
  double cost;

  if (*cost_flag == FALSE) {
    for (n = 0; n < *parameter_dimension; ++n)
      cost_tangents[n] = 2.0 * x[n];
  }

  cost = 0.0;
  for (n = 0; n < *parameter_dimension; ++n) {
    cost += (x[n] * x[n]);
  }

  *cost_flag = TRUE;

  USER_OPTIONS->Cost_Acceptance_Flag = TRUE;
  if (USER_OPTIONS->User_Acceptance_Flag == FALSE && *cost_flag == TRUE)
    USER_OPTIONS->Acceptance_Test (cost,
                                   parameter_lower_bound,
                                   parameter_upper_bound,
                                   *parameter_dimension, USER_OPTIONS);

  return (cost);
#endif /* ASA_TEMPLATE_SAMPLE */
#if MY_TEMPLATE                 /* MY_TEMPLATE_cost */
  /* Use the parameter values x[] and define your cost_function.
     The {} brackets around this function are already in place. */
#endif /* MY_TEMPLATE cost */
}
#endif /* COST_FILE */

  /* Here is a good random number generator */

#define MULT ((LONG_INT) 25173)
#define MOD ((LONG_INT) 65536)
#define INCR ((LONG_INT) 13849)
#define FMOD ((double) 65536.0)

#if ASA_LIB
/***********************************************************************
* LONG_INT asa_seed - returns initial random seed
***********************************************************************/

#if HAVE_ANSI
LONG_INT
asa_seed (LONG_INT seed)
#else
LONG_INT
asa_seed (seed)
     LONG_INT seed;
#endif
{
  static LONG_INT rand_seed;

  if (fabs ((double) seed) > 0) {
    asa_rand_seed = &rand_seed;
    rand_seed = seed;
  }

  return (rand_seed);
}
#endif /* ASA_LIB */

/***********************************************************************
* double myrand - returns random number between 0 and 1
*	This routine returns the random number generator between 0 and 1
***********************************************************************/

#if HAVE_ANSI
double
myrand (LONG_INT * rand_seed)
#else
double
myrand (rand_seed)
     LONG_INT *rand_seed;
#endif
  /* returns random number in {0,1} */
{
#if TRUE                        /* (change to FALSE for alternative RNG) */
  *rand_seed = (LONG_INT) ((MULT * (*rand_seed) + INCR) % MOD);
  return ((double) (*rand_seed) / FMOD);
#else
  /* See "Random Number Generators: Good Ones Are Hard To Find,"
     Park & Miller, CACM 31 (10) (October 1988) pp. 1192-1201.
     ***********************************************************
     THIS IMPLEMENTATION REQUIRES AT LEAST 32 BIT INTEGERS
     *********************************************************** */
#define _A_MULTIPLIER  16807L
#define _M_MODULUS     2147483647L      /* (2**31)-1 */
#define _Q_QUOTIENT    127773L  /* 2147483647 / 16807 */
#define _R_REMAINDER   2836L    /* 2147483647 % 16807 */
  long lo;
  long hi;
  long test;

  hi = *rand_seed / _Q_QUOTIENT;
  lo = *rand_seed % _Q_QUOTIENT;
  test = _A_MULTIPLIER * lo - _R_REMAINDER * hi;
  if (test > 0) {
    *rand_seed = test;
  } else {
    *rand_seed = test + _M_MODULUS;
  }
  return ((double) *rand_seed / _M_MODULUS);
#endif /* alternative RNG */
}

/***********************************************************************
* double randflt
***********************************************************************/

#if HAVE_ANSI
double
randflt (LONG_INT * rand_seed)
#else
double
randflt (rand_seed)
     LONG_INT *rand_seed;
#endif
{
  return (resettable_randflt (rand_seed, 0));
}

/***********************************************************************
* double resettable_randflt
***********************************************************************/

#if HAVE_ANSI
double
resettable_randflt (LONG_INT * rand_seed, int reset)
#else
double
resettable_randflt (rand_seed, reset)
     LONG_INT *rand_seed;
     int reset;
#endif
  /* shuffles random numbers in random_array[SHUFFLE] array */
{

  /* This RNG is a modified algorithm of that presented in
   * %A K. Binder
   * %A D. Stauffer
   * %T A simple introduction to Monte Carlo simulations and some
   *    specialized topics
   * %B Applications of the Monte Carlo Method in statistical physics
   * %E K. Binder
   * %I Springer-Verlag
   * %C Berlin
   * %D 1985
   * %P 1-36
   * where it is stated that such algorithms have been found to be
   * quite satisfactory in many statistical physics applications. */

  double rranf;
  unsigned kranf;
  int n;
  static int initial_flag = 0;
  LONG_INT initial_seed;
#if ASA_SAVE
  /* random_array[] local to all of asa_usr.c set at top of file */
#else
  static double random_array[SHUFFLE];  /* random variables */
#endif

  if (*rand_seed < 0)
    *rand_seed = -*rand_seed;

  if ((initial_flag == 0) || reset) {
    initial_seed = *rand_seed;

    for (n = 0; n < SHUFFLE; ++n)
      random_array[n] = myrand (&initial_seed);

    initial_flag = 1;

    for (n = 0; n < 1000; ++n)  /* warm up random generator */
      rranf = randflt (&initial_seed);

    rranf = randflt (rand_seed);

    return (rranf);
  }

  kranf = (unsigned) (myrand (rand_seed) * SHUFFLE) % SHUFFLE;
  rranf = *(random_array + kranf);
  *(random_array + kranf) = myrand (rand_seed);

  return (rranf);
}

#if USER_COST_SCHEDULE
#if HAVE_ANSI
double
user_cost_schedule (double test_temperature, const void *OPTIONS_TMP)
#else
double
user_cost_schedule (test_temperature, OPTIONS_TMP)
     double test_temperature;
     void *OPTIONS_TMP;
#endif /* HAVE_ANSI */
{
  double x;
  USER_DEFINES *USER_OPTIONS;

  USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

  x = 0;                        /* initialize  to prevent warning */
#if ASA_TEMPLATE_SAMPLE
  x = F_POW (test_temperature, 0.15);
#endif
#if ASA_TEMPLATE
  x = test_temperature;
#endif

  return (x);
}
#endif /* USER_COST_SCHEDULE */

#if USER_ACCEPTANCE_TEST
#if HAVE_ANSI
void
user_acceptance_test (double current_cost,
                      double *parameter_lower_bound,
                      double *parameter_upper_bound,
                      ALLOC_INT * parameter_dimension,
                      const void *OPTIONS_TMP)
#else
void
user_acceptance_test (current_cost, parameter_lower_bound,
                      parameter_upper_bound, parameter_dimension, OPTIONS_TMP)
     double current_cost;
     double *parameter_lower_bound;
     double *parameter_upper_bound;
     ALLOC_INT *parameter_dimension;
     void *OPTIONS_TMP;
#endif /* HAVE_ANSI */
{
  double uniform_test, curr_cost_temp;
  USER_DEFINES *USER_OPTIONS;

  USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

#if USER_ACCEPT_ASYMP_EXP
  double x, q, delta_cost;
#endif

#if ASA_TEMPLATE                /* ASA cost index */
  /* Calculate the current ASA cost index.  This could be useful
     to define a new schedule for the cost temperature, beyond
     simple changes that can be made using USER_COST_SCHEDULE. */

  int index;
  double k_temperature, quench, y;
  double xparameter_dimension;

#if QUENCH_COST
  quench = USER_OPTIONS->User_Quench_Cost_Scale[0];
#else
  quench = 1.0;
#endif /* QUENCH_COST */
  xparameter_dimension = (double) *parameter_dimension;
  for (index = 0; index < *parameter_dimension; ++index)
    if (fabs (parameter_upper_bound[index] - parameter_lower_bound[index]) <
        (double) EPS_DOUBLE)
      xparameter_dimension -= 1.0;

  y = -F_LOG (USER_OPTIONS->Cost_Temp_Curr
              / USER_OPTIONS->Cost_Temp_Init) / USER_OPTIONS->Cost_Temp_Scale;

  k_temperature = F_POW (y, xparameter_dimension / quench);
#endif /* ASA cost index */

  uniform_test = randflt (USER_OPTIONS->Random_Seed);
  curr_cost_temp = USER_OPTIONS->Cost_Temp_Curr;

#if ASA_TEMPLATE
#if USER_COST_SCHEDULE
  curr_cost_temp =
    (USER_OPTIONS->Cost_Schedule (USER_OPTIONS->Cost_Temp_Curr,
                                  USER_OPTIONS) + (double) EPS_DOUBLE);
#else
  curr_cost_temp = USER_OPTIONS->Cost_Temp_Curr;
#endif
#endif /* ASA_TEMPLATE */

  /* You must add in your own test here.  If USER_ACCEPT_ASYMP_EXP
     also is TRUE here, then you can use the default
     Asymp_Exp_Param=1 to replicate the code in asa.c. */

#if USER_ACCEPT_ASYMP_EXP
#if USER_COST_SCHEDULE
  curr_cost_temp =
    (USER_OPTIONS->Cost_Schedule (USER_OPTIONS->Cost_Temp_Curr,
                                  USER_OPTIONS) + (double) EPS_DOUBLE);
#endif

  delta_cost = (current_cost - *(USER_OPTIONS->Last_Cost))
    / (curr_cost_temp + (double) EPS_DOUBLE);

  /* The following asymptotic approximation to the exponential
   * function, "Tsallis statistics," was proposed in
   * %A T.J.P. Penna
   * %T Traveling salesman problem and Tsallis statistics
   * %J Phys. Rev. E
   * %V 50
   * %N 6
   * %P R1-R3
   * %D 1994
   * While the use of the TSP for a test case is of dubious value (since
   * there are many special algorithms for this problem), the use of this
   * function is another example of how to control the rate of annealing
   * of the acceptance criteria.  E.g., if you require a more moderate
   * acceptance test, then negative q may be helpful. */

  q = USER_OPTIONS->Asymp_Exp_Param;
  if (fabs (1.0 - q) < (double) EPS_DOUBLE)
    x = MIN (1.0, (F_EXP (-delta_cost)));       /* Boltzmann test */
  else if ((1.0 - (1.0 - q) * delta_cost) < (double) EPS_DOUBLE)
    x = MIN (1.0, (F_EXP (-delta_cost)));       /* Boltzmann test */
  else
    x = MIN (1.0, F_POW ((1.0 - (1.0 - q) * delta_cost), (1.0 / (1.0 - q))));

  USER_OPTIONS->Prob_Bias = x;
  if (x >= uniform_test)
    USER_OPTIONS->User_Acceptance_Flag = TRUE;
  else
    USER_OPTIONS->User_Acceptance_Flag = FALSE;

#endif /* USER_ACCEPT_ASYMP_EXP */
}
#endif /* USER_ACCEPTANCE_TEST */

#if USER_GENERATING_FUNCTION
#if HAVE_ANSI
double
user_generating_distrib (LONG_INT * seed,
                         ALLOC_INT * parameter_dimension,
                         ALLOC_INT index_v,
                         double temperature_v,
                         double init_param_temp_v,
                         double temp_scale_params_v,
                         double parameter_v,
                         double parameter_range_v,
                         double *last_saved_parameter,
                         const void *OPTIONS_TMP)
#else
double
user_generating_distrib (seed,
                         parameter_dimension,
                         index_v,
                         temperature_v,
                         init_param_temp_v,
                         temp_scale_params_v,
                         parameter_v,
                         parameter_range_v, last_saved_parameter, OPTIONS_TMP)
     LONG_INT *seed;
     ALLOC_INT *parameter_dimension;
     ALLOC_INT index_v;
     double temperature_v;
     double init_param_temp_v;
     double temp_scale_params_v;
     double parameter_v;
     double parameter_range_v;
     double *last_saved_parameter;
     void *OPTIONS_TMP;
#endif
{
  double x;
  USER_DEFINES *USER_OPTIONS;

  USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

  x = 0;                        /* initialize to prevent warning */
#if ASA_TEMPLATE
  double y, z;

  /* This is the ASA distribution.  A slower temperature schedule can be
     obtained here, e.g., temperature_v = pow(temperature_v, 0.5); */

  x = randflt (seed);
  y = x < 0.5 ? -1.0 : 1.0;
  z = y * temperature_v * (F_POW ((1.0 + 1.0 / temperature_v),
                                  fabs (2.0 * x - 1.0)) - 1.0);

  x = parameter_v + z * parameter_range_v;

#endif /* ASA_TEMPLATE */
  return (x);                   /* example return */
}
#endif /* USER_GENERATING_FUNCTION */

#if USER_REANNEAL_COST
#if HAVE_ANSI
int
user_reanneal_cost (double *cost_best,
                    double *cost_last,
                    double *initial_cost_temperature,
                    double *current_cost_temperature, const void *OPTIONS_TMP)
#else
int
user_reanneal_cost (cost_best,
                    cost_last,
                    initial_cost_temperature,
                    current_cost_temperature, OPTIONS_TMP)
     double *cost_best;
     double *cost_last;
     double *initial_cost_temperature;
     double *current_cost_temperature;
     void *OPTIONS_TMP;
#endif /* HAVE_ANSI */
{
  int cost_test;
  USER_DEFINES *USER_OPTIONS;

  USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

  cost_test = 0;                /* initialize to prevent warning */
#if ASA_TEMPLATE
  double tmp_dbl;

  static int first_time = 1;
  static double save_last[3];
  double average_cost_last;

  if (first_time == 1) {
    first_time = 0;
    save_last[0] = save_last[1] = save_last[2] = *cost_last;
  }

  save_last[2] = save_last[1];
  save_last[1] = save_last[0];
  save_last[0] = *cost_last;
  average_cost_last =
    fabs ((save_last[0] + save_last[1] + save_last[2]) / 3.0);

  tmp_dbl = MAX (fabs (*cost_best), average_cost_last);
  tmp_dbl = MAX ((double) EPS_DOUBLE, tmp_dbl);
  *initial_cost_temperature = MIN (*initial_cost_temperature, tmp_dbl);

  /* This test can be useful if your cost function goes from a positive
     to a negative value, and you do not want to get get stuck in a local
     minima around zero due to the default in reanneal().  Pick any
     number instead of 0.0001 */
  tmp_dbl = MIN (fabs (*cost_last), fabs (*cost_best));
  if (tmp_dbl < 0.0001)
    cost_test = FALSE;
  else
    cost_test = TRUE;

  /* Alternative ASA_TEMPLATE */
  tmp_dbl = MAX (fabs (*cost_last), fabs (*cost_best));
  tmp_dbl = MAX ((double) EPS_DOUBLE, tmp_dbl);
  *initial_cost_temperature = MIN (*initial_cost_temperature, tmp_dbl);

  *current_cost_temperature =
    MAX (fabs (*cost_last - *cost_best), *current_cost_temperature);
  *current_cost_temperature =
    MAX ((double) EPS_DOUBLE, *current_cost_temperature);
  *current_cost_temperature =
    MIN (*current_cost_temperature, *initial_cost_temperature);

  cost_test = TRUE;
#endif /* ASA_TEMPLATE */

  return (cost_test);           /* example return */
}
#endif /* USER_REANNEAL_COST */

#if USER_REANNEAL_PARAMETERS
#if HAVE_ANSI
double
user_reanneal_params (double current_temp,
                      double tangent,
                      double max_tangent, const void *OPTIONS_TMP)
#else
double
user_reanneal_params (current_temp, tangent, max_tangent, OPTIONS_TMP)
     double current_temp;
     double tangent;
     double max_tangent;
     void *OPTIONS_TMP;
#endif /* HAVE_ANSI */
{
  double x;
  USER_DEFINES *USER_OPTIONS;

  USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

  x = 0;                        /* initialize to prevent warning */
#if ASA_TEMPLATE

  x = current_temp * (max_tangent / tangent);

#endif
  return (x);                   /* example return */
}
#endif /* USER_REANNEAL_PARAMETERS */

#if SELF_OPTIMIZE

/***********************************************************************
* main
*	This is a sample calling program to self-optimize ASA
***********************************************************************/
#if HAVE_ANSI

#if ASA_LIB
int
asa_main (
#if ASA_TEMPLATE_LIB
           double *main_recur_cost_value,
           double *main_recur_cost_parameters, int *main_recur_exit_code
#endif
  )
#else /* ASA_LIB */
int
main (int argc, char **argv)
#endif                          /* ASA_LIB */
#else /* HAVE_ANSI */

#if ASA_LIB
int
asa_main (
#if ASA_TEMPLATE_LIB
           main_recur_cost_value,
           main_recur_cost_parameters, main_recur_exit_code
#endif
  )
#if ASA_TEMPLATE_LIB
     double *main_recur_cost_value;
     double *main_recur_cost_parameters;
     int *main_recur_exit_code;
#endif

#else /* ASA_LIB */
int
main (argc, argv)
     int argc;
     char **argv;
#endif /* ASA_LIB */

#endif /* HAVE_ANSI */
{
  /* seed for random number generator */
  LONG_INT *recur_rand_seed;
  char user_exit_msg[160];
  FILE *ptr_out;

#if RECUR_OPTIONS_FILE
  int fscanf_ret;
  FILE *recur_ptr_options;
  char read_option[80];
  char read_if[4], read_FALSE[6], read_comm1[3], read_ASA_SAVE[9],
    read_comm2[3];
  int read_int;
#if INT_LONG
  LONG_INT read_long;
#endif
  double read_double;
#endif /* RECUR_OPTIONS_FILE */

  int *recur_exit_code;
#if ASA_LIB
#else
  int compile_cnt;
#endif
#if MULTI_MIN
  int multi_index;
  ALLOC_INT n_param;
#endif

  double *recur_parameter_lower_bound, *recur_parameter_upper_bound;
  double *recur_cost_parameters, *recur_cost_tangents, *recur_cost_curvature;
  double recur_cost_value;

  ALLOC_INT *recur_parameter_dimension;
  int *recur_parameter_int_real;
  int *recur_cost_flag;
  int recur_initialize_params_value;
  ALLOC_INT recur_v;
#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_main_decl */
  /* add some declarations if required */
#endif

  USER_DEFINES *RECUR_USER_OPTIONS;

  if ((recur_parameter_dimension =
       (ALLOC_INT *) calloc (1, sizeof (ALLOC_INT))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): recur_parameter_dimension");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  if ((recur_exit_code = (int *) calloc (1, sizeof (int))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): recur_exit_code");
    Exit_USER (user_exit_msg);
    free (recur_parameter_dimension);
    return (-2);
  }
  if ((recur_cost_flag = (int *) calloc (1, sizeof (int))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): recur_cost_flag");
    Exit_USER (user_exit_msg);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    return (-2);
  }

  if ((RECUR_USER_OPTIONS =
       (USER_DEFINES *) calloc (1, sizeof (USER_DEFINES))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): RECUR_USER_OPTIONS");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    return (-2);
  }
#if RECUR_OPTIONS_FILE
  recur_ptr_options = fopen ("asa_opt_recur", "r");

  fscanf_ret = fscanf (recur_ptr_options, "%s%s%s%s%s",
                       read_if, read_FALSE, read_comm1, read_ASA_SAVE,
                       read_comm2);
  if (strcmp (read_if, "#if") || strcmp (read_FALSE, "FALSE")
      || strcmp (read_comm1, "/*") || strcmp (read_ASA_SAVE, "ASA_SAVE")
      || strcmp (read_comm2, "*/")) {
#if INCL_STDOUT
    printf ("\n\n*** EXIT not asa_opt_recur for this version *** \n\n");
#endif /* INCL_STDOUT */
    free (RECUR_USER_OPTIONS);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
#if RECUR_OPTIONS_FILE
    fclose (recur_ptr_options);
#endif
    return (-6);
  }
#if INT_LONG
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%ld", &read_long);
  RECUR_USER_OPTIONS->Limit_Acceptances = read_long;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%ld", &read_long);
  RECUR_USER_OPTIONS->Limit_Generated = read_long;
#else
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Limit_Acceptances = read_int;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Limit_Generated = read_int;
#endif
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Limit_Invalid_Generated_States = read_int;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%lf", &read_double);
  RECUR_USER_OPTIONS->Accepted_To_Generated_Ratio = read_double;

  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%lf", &read_double);
  RECUR_USER_OPTIONS->Cost_Precision = read_double;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Maximum_Cost_Repeat = read_int;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Number_Cost_Samples = read_int;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%lf", &read_double);
  RECUR_USER_OPTIONS->Temperature_Ratio_Scale = read_double;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%lf", &read_double);
  RECUR_USER_OPTIONS->Cost_Parameter_Scale_Ratio = read_double;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%lf", &read_double);
  RECUR_USER_OPTIONS->Temperature_Anneal_Scale = read_double;

  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Include_Integer_Parameters = read_int;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->User_Initial_Parameters = read_int;
#if INT_ALLOC
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Sequential_Parameters = read_int;
#else
#if INT_LONG
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%ld", &read_long);
  RECUR_USER_OPTIONS->Sequential_Parameters = read_long;
#else
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Sequential_Parameters = read_int;
#endif
#endif
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%lf", &read_double);
  RECUR_USER_OPTIONS->Initial_Parameter_Temperature = read_double;

  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Acceptance_Frequency_Modulus = read_int;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Generated_Frequency_Modulus = read_int;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Reanneal_Cost = read_int;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Reanneal_Parameters = read_int;

  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%lf", &read_double);
  RECUR_USER_OPTIONS->Delta_X = read_double;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->User_Tangents = read_int;
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  RECUR_USER_OPTIONS->Curvature_0 = read_int;

#else /* RECUR_OPTIONS_FILE */
  RECUR_USER_OPTIONS->Limit_Acceptances = 100;
  RECUR_USER_OPTIONS->Limit_Generated = 1000;
  RECUR_USER_OPTIONS->Limit_Invalid_Generated_States = 1000;
  RECUR_USER_OPTIONS->Accepted_To_Generated_Ratio = 1.0E-4;

  RECUR_USER_OPTIONS->Cost_Precision = 1.0E-18;
  RECUR_USER_OPTIONS->Maximum_Cost_Repeat = 2;
  RECUR_USER_OPTIONS->Number_Cost_Samples = 2;
  RECUR_USER_OPTIONS->Temperature_Ratio_Scale = 1.0E-5;
  RECUR_USER_OPTIONS->Cost_Parameter_Scale_Ratio = 1.0;
  RECUR_USER_OPTIONS->Temperature_Anneal_Scale = 100.0;

  RECUR_USER_OPTIONS->Include_Integer_Parameters = FALSE;
  RECUR_USER_OPTIONS->User_Initial_Parameters = TRUE;
  RECUR_USER_OPTIONS->Sequential_Parameters = -1;
  RECUR_USER_OPTIONS->Initial_Parameter_Temperature = 1.0;

  RECUR_USER_OPTIONS->Acceptance_Frequency_Modulus = 15;
  RECUR_USER_OPTIONS->Generated_Frequency_Modulus = 10000;
  RECUR_USER_OPTIONS->Reanneal_Cost = FALSE;
  RECUR_USER_OPTIONS->Reanneal_Parameters = FALSE;

  RECUR_USER_OPTIONS->Delta_X = 1.0E-6;
  RECUR_USER_OPTIONS->User_Tangents = FALSE;
  RECUR_USER_OPTIONS->Curvature_0 = TRUE;

#endif /* RECUR_OPTIONS_FILE */

  /* the number of parameters for the recur_cost_function */
#if RECUR_OPTIONS_FILE_DATA
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);

#if INT_ALLOC
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  *recur_parameter_dimension = read_int;
#else
#if INT_LONG
  fscanf_ret = fscanf (recur_ptr_options, "%ld", &read_long);
  *recur_parameter_dimension = read_long;
#else
  fscanf_ret = fscanf (recur_ptr_options, "%d", &read_int);
  *recur_parameter_dimension = read_int;
#endif
#endif

#else /* RECUR_OPTIONS_FILE_DATA */
#if ASA_TEMPLATE_SELFOPT
  *recur_parameter_dimension = 2;
#endif
#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_dim */
  /* If not using RECUR_OPTIONS_FILE_DATA or data read from recur_asa_opt,
     insert the number of parameters for the recur_cost_function */
#endif /* MY_TEMPLATE recur_dim */
#endif /* RECUR_OPTIONS_FILE_DATA */
  if ((recur_parameter_lower_bound =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): recur_parameter_lower_bound");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
#if RECUR_OPTIONS_FILE
    fclose (recur_ptr_options);
#endif
    return (-2);
  }
  if ((recur_parameter_upper_bound =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): recur_parameter_upper_bound");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
    free (recur_parameter_lower_bound);
#if RECUR_OPTIONS_FILE
    fclose (recur_ptr_options);
#endif
    return (-2);
  }

  if ((recur_cost_parameters =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): recur_cost_parameters");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
    free (recur_parameter_lower_bound);
    free (recur_parameter_upper_bound);
#if RECUR_OPTIONS_FILE
    fclose (recur_ptr_options);
#endif
    return (-2);
  }

  if ((recur_parameter_int_real =
       (int *) calloc (*recur_parameter_dimension, sizeof (int))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): recur_parameter_int_real");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
    free (recur_parameter_lower_bound);
    free (recur_parameter_upper_bound);
    free (recur_cost_parameters);
#if RECUR_OPTIONS_FILE
    fclose (recur_ptr_options);
#endif
    return (-2);
  }

  if ((recur_cost_tangents =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): recur_cost_tangents");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
    free (recur_parameter_lower_bound);
    free (recur_parameter_upper_bound);
    free (recur_cost_parameters);
    free (recur_parameter_int_real);
#if RECUR_OPTIONS_FILE
    fclose (recur_ptr_options);
#endif
    return (-2);
  }

  if (RECUR_USER_OPTIONS->Curvature_0 == FALSE
      || RECUR_USER_OPTIONS->Curvature_0 == -1) {

    if ((recur_cost_curvature =
         (double *) calloc ((*recur_parameter_dimension)
                            * (*recur_parameter_dimension),
                            sizeof (double))) == NULL) {
      strcpy (user_exit_msg, "main()/asa_main(): recur_cost_curvature");
      Exit_USER (user_exit_msg);
      free (recur_cost_flag);
      free (recur_exit_code);
      free (recur_parameter_dimension);
      free (RECUR_USER_OPTIONS);
      free (recur_parameter_lower_bound);
      free (recur_parameter_upper_bound);
      free (recur_cost_parameters);
      free (recur_parameter_int_real);
      free (recur_cost_tangents);
#if RECUR_OPTIONS_FILE
      fclose (recur_ptr_options);
#endif
      return (-2);
    }
  } else {
    recur_cost_curvature = (double *) NULL;
  }

#if ASA_TEMPLATE_SELFOPT
  /* Set memory to that required for use. */
  RECUR_USER_OPTIONS->Asa_Data_Dim_Dbl = 1;
  if ((RECUR_USER_OPTIONS->Asa_Data_Dbl =
       (double *) calloc (RECUR_USER_OPTIONS->Asa_Data_Dim_Dbl,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "main()/asa_main(): RECUR_USER_OPTIONS->Asa_Data_Dbl");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
    free (recur_parameter_lower_bound);
    free (recur_parameter_upper_bound);
    free (recur_cost_parameters);
    free (recur_parameter_int_real);
    free (recur_cost_tangents);
    free (recur_cost_curvature);
#if RECUR_OPTIONS_FILE
    fclose (recur_ptr_options);
#endif
    return (-2);
  }
  /* Use Asa_Data[0] as flag, e.g., if used with SELF_OPTIMIZE. */
  RECUR_USER_OPTIONS->Asa_Data_Dbl[0] = 0;
#endif /* ASA_TEMPLATE_SELFOPT */

#if OPTIONAL_DATA_PTR
#if ASA_TEMPLATE
  /* see note at "Instead of freeing Asa_Data_Ptr" */
  USER_OPTIONS->Asa_Data_Ptr = OptionalPointer;
  RECUR_USER_OPTIONS->Asa_Data_Dim_Ptr = 1;
  if ((RECUR_USER_OPTIONS->Asa_Data_Ptr =
       (OPTIONAL_PTR_TYPE *) calloc (RECUR_USER_OPTIONS->Asa_Data_Dim_Ptr,
                                     sizeof (OPTIONAL_PTR_TYPE))) == NULL) {
    strcpy (user_exit_msg,
            "main()/asa_main(): RECUR_USER_OPTIONS->Asa_Data_Ptr");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
    free (recur_parameter_lower_bound);
    free (recur_parameter_upper_bound);
    free (recur_cost_parameters);
    free (recur_parameter_int_real);
    free (recur_cost_tangents);
    free (recur_cost_curvature);
#if RECUR_OPTIONS_FILE
    fclose (recur_ptr_options);
#endif
    return (-2);
  }
#endif /* ASA_TEMPLATE */
#endif /* OPTIONAL_DATA_PTR */

#if ASA_SAVE
  /* Such data could be saved in a user_save file, but for
     convenience here everything is saved in asa_save. */
  RECUR_USER_OPTIONS->Random_Array_Dim = SHUFFLE;
  RECUR_USER_OPTIONS->Random_Array = random_array;
#endif /* ASA_SAVE */

  /* open the output file */
#if ASA_SAVE
  if (!strcmp (USER_OUT, "STDOUT")) {
#if INCL_STDOUT
    ptr_out = stdout;
#endif /* INCL_STDOUT */
  } else {
    ptr_out = fopen (USER_OUT, "a");
  }
#else
  if (!strcmp (USER_OUT, "STDOUT")) {
#if INCL_STDOUT
    ptr_out = stdout;
#endif /* INCL_STDOUT */
  } else {
    ptr_out = fopen (USER_OUT, "w");
  }
#endif
  fprintf (ptr_out, "%s\n\n", USER_ID);

#if ASA_LIB
#else
  /* print out compile options set by user in Makefile */
  if (argc > 1) {
    fprintf (ptr_out, "CC = %s\n", argv[1]);
    for (compile_cnt = 2; compile_cnt < argc; ++compile_cnt) {
      fprintf (ptr_out, "\t%s\n", argv[compile_cnt]);
    }
    fprintf (ptr_out, "\n");
  }
#endif
#if TIME_CALC
  /* print starting time */
  print_time ("start", ptr_out);
#endif
  fflush (ptr_out);

  if ((recur_rand_seed =
       (ALLOC_INT *) calloc (1, sizeof (ALLOC_INT))) == NULL) {
    strcpy (user_exit_msg, "main()/asa_main(): recur_rand_seed");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
    free (recur_parameter_lower_bound);
    free (recur_parameter_upper_bound);
    free (recur_cost_parameters);
    free (recur_parameter_int_real);
    free (recur_cost_tangents);
    free (recur_cost_curvature);
#if RECUR_OPTIONS_FILE
    fclose (recur_ptr_options);
#endif
    return (-2);
  }

  /* first value of *recur_rand_seed */
#if ASA_LIB
  *recur_rand_seed = (asa_rand_seed ? *asa_rand_seed : (LONG_INT) 696969);
#else
  *recur_rand_seed = 696969;
#endif

  randflt (recur_rand_seed);

#if USER_COST_SCHEDULE
  RECUR_USER_OPTIONS->Cost_Schedule = recur_user_cost_schedule;
#endif
#if USER_ACCEPTANCE_TEST
  RECUR_USER_OPTIONS->Acceptance_Test = recur_user_acceptance_test;
#endif
#if USER_ACCEPT_ASYMP_EXP
  RECUR_USER_OPTIONS->Asymp_Exp_Param = 1.0;
#endif
#if USER_GENERATING_FUNCTION
  RECUR_USER_OPTIONS->Generating_Distrib = recur_user_generating_distrib;
#endif
#if USER_REANNEAL_COST
  RECUR_USER_OPTIONS->Reanneal_Cost_Function = recur_user_reanneal_cost;
#endif
#if USER_REANNEAL_PARAMETERS
  RECUR_USER_OPTIONS->Reanneal_Params_Function = recur_user_reanneal_params;
#endif

#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_pre_initialize */
  /* last changes before entering recur_initialize_parameters() */
#endif

  /* initialize the users parameters, allocating space, etc.
     Note that the default is to have asa generate the initial
     recur_cost_parameters that satisfy the user's constraints. */

  recur_initialize_params_value =
    recur_initialize_parameters (recur_cost_parameters,
                                 recur_parameter_lower_bound,
                                 recur_parameter_upper_bound,
                                 recur_cost_tangents,
                                 recur_cost_curvature,
                                 recur_parameter_dimension,
                                 recur_parameter_int_real,
#if RECUR_OPTIONS_FILE_DATA
                                 recur_ptr_options,
#endif
                                 RECUR_USER_OPTIONS);
#if RECUR_OPTIONS_FILE
  fclose (recur_ptr_options);
#endif
  if (recur_initialize_params_value == -2) {
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
    free (recur_parameter_lower_bound);
    free (recur_parameter_upper_bound);
    free (recur_cost_parameters);
    free (recur_parameter_int_real);
    free (recur_cost_tangents);
    free (recur_cost_curvature);
    return (recur_initialize_params_value);
  }
#if USER_ASA_USR_OUT
  if ((RECUR_USER_OPTIONS->Asa_Usr_Out_File =
       (char *) calloc (80, sizeof (char))) == NULL) {
    strcpy (user_exit_msg,
            "main()/asa_main(): RECUR_USER_OPTIONS->Asa_Usr_Out_File");
  }
#endif
#if USER_ASA_OUT
  if ((RECUR_USER_OPTIONS->Asa_Out_File =
       (char *) calloc (80, sizeof (char))) == NULL) {
    strcpy (user_exit_msg,
            "main()/asa_main(): RECUR_USER_OPTIONS->Asa_Out_File");
    Exit_USER (user_exit_msg);
    free (recur_cost_flag);
    free (recur_exit_code);
    free (recur_parameter_dimension);
    free (RECUR_USER_OPTIONS);
    free (recur_parameter_lower_bound);
    free (recur_parameter_upper_bound);
    free (recur_cost_parameters);
    free (recur_parameter_int_real);
    free (recur_cost_tangents);
    free (recur_cost_curvature);
    return (-2);
  }
#if ASA_TEMPLATE_SELFOPT
  strcpy (RECUR_USER_OPTIONS->Asa_Out_File, "asa_sfop");
#endif
#endif

#if ASA_FUZZY
  /* can use in inner optimization shell */
  InitFuzzyASA (USER_OPTIONS, *recur_parameter_dimension);
#endif /* ASA_FUZZY */

  recur_cost_value = asa (RECUR_USER_COST_FUNCTION,
                          randflt,
                          recur_rand_seed,
                          recur_cost_parameters,
                          recur_parameter_lower_bound,
                          recur_parameter_upper_bound,
                          recur_cost_tangents,
                          recur_cost_curvature,
                          recur_parameter_dimension,
                          recur_parameter_int_real,
                          recur_cost_flag,
                          recur_exit_code, RECUR_USER_OPTIONS);
  if (*recur_exit_code == -1) {
#if INCL_STDOUT
    printf ("\n\n*** error in calloc in ASA ***\n\n");
#endif /* INCL_STDOUT */
    fprintf (ptr_out, "\n\n*** error in calloc in ASA ***\n\n");
    fflush (ptr_out);
    return (-1);
  }
#if ASA_FUZZY
  if (USER_OPTIONS->Locate_Cost == 12) {
    USER_OPTIONS->Locate_Cost = 0;
  }
  CloseFuzzyASA (RECUR_USER_OPTIONS);
#endif /* ASA_FUZZY */
#if MULTI_MIN
  fprintf (ptr_out, "Multi_Specify = %d\n",
           RECUR_USER_OPTIONS->Multi_Specify);
  for (n_param = 0; n_param < *recur_parameter_dimension; ++n_param) {
    fprintf (ptr_out,
#if INT_ALLOC
             "Multi_Grid[%d] = %12.7g\n",
#else
#if INT_LONG
             "Multi_Grid[%ld] = %12.7g\n",
#else
             "Multi_Grid[%d] = %12.7g\n",
#endif
#endif
             n_param, RECUR_USER_OPTIONS->Multi_Grid[n_param]);
  }
  fprintf (ptr_out, "\n");
  for (multi_index = 0; multi_index < RECUR_USER_OPTIONS->Multi_Number;
       ++multi_index) {
    fprintf (ptr_out, "\n");
    fprintf (ptr_out, "Multi_Cost[%d] = %12.7g\n",
             multi_index, RECUR_USER_OPTIONS->Multi_Cost[multi_index]);
    for (n_param = 0; n_param < *recur_parameter_dimension; ++n_param) {
      fprintf (ptr_out,
#if INT_ALLOC
               "Multi_Params[%d][%d] = %12.7g\n",
#else
#if INT_LONG
               "Multi_Params[%d][%ld] = %12.7g\n",
#else
               "Multi_Params[%d][%d] = %12.7g\n",
#endif
#endif
               multi_index, n_param,
               RECUR_USER_OPTIONS->Multi_Params[multi_index][n_param]);
    }
  }
  fprintf (ptr_out, "\n");
  fflush (ptr_out);
#endif /* MULTI_MIN */

#if FITLOC
  /* Fit_Local and Penalty may be set adaptively */
  RECUR_USER_OPTIONS->Penalty = 1000;
  RECUR_USER_OPTIONS->Fit_Local = 1;
  RECUR_USER_OPTIONS->Iter_Max = 500;
  if (RECUR_USER_OPTIONS->Fit_Local >= 1) {
    recur_cost_value = fitloc (RECUR_USER_COST_FUNCTION,
                               recur_cost_parameters,
                               recur_parameter_lower_bound,
                               recur_parameter_upper_bound,
                               recur_cost_tangents,
                               recur_cost_curvature,
                               recur_parameter_dimension,
                               recur_parameter_int_real,
                               recur_cost_flag,
                               recur_exit_code, RECUR_USER_OPTIONS, ptr_out);
  }
#endif /* FITLOC */

  fprintf (ptr_out, "\n\n recur_cost_value = %12.7g\n", recur_cost_value);
#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_post_recur_asa */
#endif
#if ASA_TEMPLATE_LIB
  *main_recur_cost_value = recur_cost_value;
  for (recur_v = 0; recur_v < *recur_parameter_dimension; ++recur_v) {
    main_recur_cost_parameters[recur_v] = recur_cost_parameters[recur_v];
  }
  *main_recur_exit_code = *recur_exit_code;
#endif

  for (recur_v = 0; recur_v < *recur_parameter_dimension; ++recur_v)
#if INT_ALLOC
    fprintf (ptr_out, "recur_cost_parameters[%d] = %12.7g\n",
#else
#if INT_LONG
    fprintf (ptr_out, "recur_cost_parameters[%ld] = %12.7g\n",
#else
    fprintf (ptr_out, "recur_cost_parameters[%d] = %12.7g\n",
#endif
#endif
             recur_v, recur_cost_parameters[recur_v]);

  fprintf (ptr_out, "\n\n");

#if TIME_CALC
  /* print ending time */
  print_time ("end", ptr_out);
#endif

  /* close all files */
  fclose (ptr_out);

#if OPTIONAL_DATA_DBL
  free (RECUR_USER_OPTIONS->Asa_Data_Dbl);
#endif
#if OPTIONAL_DATA_INT
  free (RECUR_USER_OPTIONS->Asa_Data_Int);
#endif
#if OPTIONAL_DATA_PTR
  free (RECUR_USER_OPTIONS->Asa_Data_Ptr);
#endif
#if USER_ASA_OUT
#if TEMPLATE
  /* if necessary */
  free (recur_asa_out_my);
#endif
  free (RECUR_USER_OPTIONS->Asa_Out_File);
#endif
#if USER_ASA_USR_OUT
#if ASA_TEMPLATE
  /* if necessary */
  free (recur_asa_usr_out_my);
#endif
  free (RECUR_USER_OPTIONS->Asa_Usr_Out_File);
#endif
#if ASA_QUEUE
#if ASA_RESOLUTION
#else
  free (RECUR_USER_OPTIONS->Queue_Resolution);
#endif
#endif
#if ASA_RESOLUTION
  free (RECUR_USER_OPTIONS->Coarse_Resolution);
#endif
  if (RECUR_USER_OPTIONS->Curvature_0 == FALSE
      || RECUR_USER_OPTIONS->Curvature_0 == -1)
    free (recur_cost_curvature);
#if USER_INITIAL_PARAMETERS_TEMPS
  free (RECUR_USER_OPTIONS->User_Parameter_Temperature);
#endif
#if USER_INITIAL_COST_TEMP
  free (RECUR_USER_OPTIONS->User_Cost_Temperature);
#endif
#if DELTA_PARAMETERS
  free (RECUR_USER_OPTIONS->User_Delta_Parameter);
#endif
#if QUENCH_PARAMETERS
  free (RECUR_USER_OPTIONS->User_Quench_Param_Scale);
#endif
#if QUENCH_COST
  free (RECUR_USER_OPTIONS->User_Quench_Cost_Scale);
#endif
#if RATIO_TEMPERATURE_SCALES
  free (RECUR_USER_OPTIONS->User_Temperature_Ratio);
#endif
#if MULTI_MIN
  free (RECUR_USER_OPTIONS->Multi_Cost);
  free (RECUR_USER_OPTIONS->Multi_Grid);
  for (multi_index = 0; multi_index < RECUR_USER_OPTIONS->Multi_Number;
       ++multi_index) {
    free (RECUR_USER_OPTIONS->Multi_Params[multi_index]);
  }
  free (RECUR_USER_OPTIONS->Multi_Params);
#endif /* MULTI_MIN */
  free (RECUR_USER_OPTIONS);
  free (recur_parameter_dimension);
  free (recur_exit_code);
  free (recur_cost_flag);
  free (recur_parameter_lower_bound);
  free (recur_parameter_upper_bound);
  free (recur_cost_parameters);
  free (recur_parameter_int_real);
  free (recur_cost_tangents);
  free (recur_rand_seed);

  return (0);
  /* NOTREACHED */
}

/***********************************************************************
* recur_initialize_parameters
*	This depends on the users cost function to optimize (minimum).
*	The routine allocates storage needed for asa. The user should
*	define the number of parameters and their ranges,
*	and make sure the initial parameters are within
*	the minimum and maximum ranges. The array
*	recur_parameter_int_real should be REAL_TYPE (-1)
*       for real parameters,
***********************************************************************/
#if HAVE_ANSI
int
recur_initialize_parameters (double *recur_cost_parameters,
                             double *recur_parameter_lower_bound,
                             double *recur_parameter_upper_bound,
                             double *recur_cost_tangents,
                             double *recur_cost_curvature,
                             ALLOC_INT * recur_parameter_dimension,
                             int *recur_parameter_int_real,
#if RECUR_OPTIONS_FILE_DATA
                             FILE * recur_ptr_options,
#endif
                             USER_DEFINES * RECUR_USER_OPTIONS)
#else
int
recur_initialize_parameters (recur_cost_parameters,
                             recur_parameter_lower_bound,
                             recur_parameter_upper_bound,
                             recur_cost_tangents,
                             recur_cost_curvature,
                             recur_parameter_dimension,
                             recur_parameter_int_real,
#if RECUR_OPTIONS_FILE_DATA
                             recur_ptr_options,
#endif
                             RECUR_USER_OPTIONS)
     double *recur_parameter_lower_bound;
     double *recur_parameter_upper_bound;
     double *recur_cost_parameters;
     double *recur_cost_tangents;
     double *recur_cost_curvature;
     ALLOC_INT *recur_parameter_dimension;
     int *recur_parameter_int_real;
#if RECUR_OPTIONS_FILE_DATA
     FILE *recur_ptr_options;
#endif
     USER_DEFINES *RECUR_USER_OPTIONS;
#endif
{
  int fscanf_ret;
  ALLOC_INT index;
#if RECUR_OPTIONS_FILE_DATA
  char read_option[80];
  ALLOC_INT read_index;
#endif
#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_init_decl */
  /* add some declarations if required */
#endif
#if MULTI_MIN
  int multi_index;
#endif

  index = 0;                    /* initialize to prevent warning */
  fscanf_ret = 0;

#if RECUR_OPTIONS_FILE_DATA
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);

  for (index = 0; index < *recur_parameter_dimension; ++index) {
#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_read_opt */
    /* put in some code as required to alter lines read from recur_asa_opt */
#endif
#if INT_ALLOC
    fscanf_ret = fscanf (recur_ptr_options, "%d", &read_index);
#else
#if INT_LONG
    fscanf_ret = fscanf (recur_ptr_options, "%ld", &read_index);
#else
    fscanf_ret = fscanf (recur_ptr_options, "%d", &read_index);
#endif
#endif
    fscanf_ret = fscanf (recur_ptr_options, "%lf%lf%lf%d",
                         &(recur_parameter_lower_bound[read_index]),
                         &(recur_parameter_upper_bound[read_index]),
                         &(recur_cost_parameters[read_index]),
                         &(recur_parameter_int_real[read_index]));
  }
#else /* RECUR_OPTIONS_FILE_DATA */
#if ASA_TEMPLATE_SELFOPT
  /*  NOTE:
     USER_OPTIONS->Temperature_Ratio_Scale = x[0];
     USER_OPTIONS->Cost_Parameter_Scale_Ratio = x[1];
   */

  /* store the initial parameter values */
  recur_cost_parameters[0] = 1.0E-5;
  recur_cost_parameters[1] = 1.0;

  recur_parameter_lower_bound[0] = 1.0E-6;
  recur_parameter_upper_bound[0] = 1.0E-4;

  recur_parameter_lower_bound[1] = 0.5;
  recur_parameter_upper_bound[1] = 3.0;

  /* store the initial parameter types */
  for (index = 0; index < *recur_parameter_dimension; ++index)
    recur_parameter_int_real[index] = REAL_TYPE;
#endif
#endif /* RECUR_OPTIONS_FILE_DATA */

#if USER_INITIAL_PARAMETERS_TEMPS
  if ((RECUR_USER_OPTIONS->User_Parameter_Temperature =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_initialize_parameters(): RECUR_USER_OPTIONS->User_Parameter_Temperature");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  for (index = 0; index < *recur_parameter_dimension; ++index)
    RECUR_USER_OPTIONS->User_Parameter_Temperature[index] = 1.0;
#endif /* USER_INITIAL_PARAMETERS_TEMPS */
#if USER_INITIAL_COST_TEMP
  if ((RECUR_USER_OPTIONS->User_Cost_Temperature =
       (double *) calloc (1, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_initialize_parameters(): RECUR_USER_OPTIONS->User_Cost_Temperature");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  RECUR_USER_OPTIONS->User_Cost_Temperature[0] = 5.936648E+09;
#endif /* USER_INITIAL_COST_TEMP */
#if DELTA_PARAMETERS
  if ((RECUR_USER_OPTIONS->User_Delta_Parameter =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_initialize_parameters(): RECUR_USER_OPTIONS->User_Delta_Parameter");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  for (index = 0; index < *recur_parameter_dimension; ++index)
    RECUR_USER_OPTIONS->User_Delta_Parameter[index] = 0.001;
#endif /* DELTA_PARAMETERS */
#if QUENCH_PARAMETERS
  if ((RECUR_USER_OPTIONS->User_Quench_Param_Scale =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_initialize_parameters(): RECUR_USER_OPTIONS->User_Quench_Param_Scale");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  for (index = 0; index < *recur_parameter_dimension; ++index)
    RECUR_USER_OPTIONS->User_Quench_Param_Scale[index] = 1.0;
#endif
#endif /* QUENCH_PARAMETERS */
#if QUENCH_COST
  if ((RECUR_USER_OPTIONS->User_Quench_Cost_Scale =
       (double *) calloc (1, sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_initialize_parameters(): RECUR_USER_OPTIONS->User_Quench_Cost_Scale");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  RECUR_USER_OPTIONS->User_Quench_Cost_Scale[0] = 1.0;
#endif
#endif /* QUENCH_COST */

  /* use asa_opt_recur to read in QUENCH RECUR_USER_OPTIONS */
#if RECUR_OPTIONS_FILE_DATA
#if QUENCH_COST
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%lf",
                       &(RECUR_USER_OPTIONS->User_Quench_Cost_Scale[0]));

#if QUENCH_PARAMETERS
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  fscanf_ret = fscanf (recur_ptr_options, "%s", read_option);
  for (index = 0; index < *recur_parameter_dimension; ++index) {
#if INT_ALLOC
    fscanf_ret = fscanf (recur_ptr_options, "%d", &read_index);
#else
#if INT_LONG
    fscanf_ret = fscanf (recur_ptr_options, "%ld", &read_index);
#else
    fscanf_ret = fscanf (recur_ptr_options, "%d", &read_index);
#endif
#endif
    fscanf_ret = fscanf (recur_ptr_options, "%lf",
                         &(RECUR_USER_OPTIONS->User_Quench_Param_Scale
                           [read_index]));
  }
#endif /* QUENCH_PARAMETERS */
#endif /* QUENCH_COST */
#endif /* RECUR_OPTIONS_FILE_DATA */
#if RATIO_TEMPERATURE_SCALES
  if ((RECUR_USER_OPTIONS->User_Temperature_Ratio =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_initialize_parameters(): RECUR_USER_OPTIONS->User_Temperature_Ratio");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if ASA_TEMPLATE
  for (index = 0; index < *recur_parameter_dimension; ++index)
    RECUR_USER_OPTIONS->User_Temperature_Ratio[index] = 1.0;
#endif
#endif /* RATIO_TEMPERATURE_SCALES */
  /* Defines the limit of collection of sampled data by asa */

#if ASA_TEMPLATE
#if ASA_PARALLEL
  RECUR_USER_OPTIONS->Gener_Block = 1;
  RECUR_USER_OPTIONS->Gener_Block_Max = 1;
  RECUR_USER_OPTIONS->Gener_Mov_Avr = 1;
#endif
#endif
#if ASA_RESOLUTION
  if ((RECUR_USER_OPTIONS->Coarse_Resolution =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_initialize_parameters(): RECUR_USER_OPTIONS->Coarse_Resolution");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#endif
#if MULTI_MIN
#if ASA_TEMPLATE
  RECUR_USER_OPTIONS->Multi_Number = 2;
#endif
  if ((RECUR_USER_OPTIONS->Multi_Cost =
       (double *) calloc (RECUR_USER_OPTIONS->Multi_Number,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): RECUR_USER_OPTIONS->Multi_Cost");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  if ((RECUR_USER_OPTIONS->Multi_Grid =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_initialize_parameters(): RECUR_USER_OPTIONS->Multi_Grid");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  if ((RECUR_USER_OPTIONS->Multi_Params =
       (double **) calloc (RECUR_USER_OPTIONS->Multi_Number,
                           sizeof (double *))) == NULL) {
    strcpy (user_exit_msg,
            "initialize_parameters(): RECUR_USER_OPTIONS->Multi_Params");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  for (multi_index = 0; multi_index < RECUR_USER_OPTIONS->Multi_Number;
       ++multi_index) {
    if ((RECUR_USER_OPTIONS->Multi_Params[multi_index] =
         (double *) calloc (*recur_parameter_dimension,
                            sizeof (double))) == NULL) {
      strcpy (user_exit_msg,
              "recur_initialize_parameters(): RECUR_USER_OPTIONS->Multi_Params[multi_index]");
      Exit_USER (user_exit_msg);
      return (-2);
    }
  }
#if ASA_TEST
  for (index = 0; index < *recur_parameter_dimension; ++index) {
    RECUR_USER_OPTIONS->Multi_Grid[index] = 0.05;
  }
  RECUR_USER_OPTIONS->Multi_Specify = 0;
#endif
#if ASA_TEMPLATE
  for (index = 0; index < *recur_parameter_dimension; ++index) {
    RECUR_USER_OPTIONS->Multi_Grid[index] =
      (recur_parameter_upper_bound[index] -
       recur_parameter_lower_bound[index]) / 100.0;
  }
  RECUR_USER_OPTIONS->Multi_Specify = 0;
#endif /* ASA_TEMPLATE */
#endif /* MULTI_MIN */
#if ASA_TEMPLATE_QUEUE
  RECUR_USER_OPTIONS->Queue_Size = 0;
#endif
#if ASA_QUEUE
#if ASA_RESOLUTION
  RECUR_USER_OPTIONS->Queue_Resolution =
    RECUR_USER_OPTIONS->Coarse_Resolution;
#else /* ASA_RESOLUTION */
  if ((RECUR_USER_OPTIONS->Queue_Resolution =
       (double *) calloc (*recur_parameter_dimension,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_initialize_parameters(): RECUR_USER_OPTIONS->Queue_Resolution");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#endif /* ASA_RESOLUTION */
#if ASA_TEMPLATE_QUEUE
  RECUR_USER_OPTIONS->Queue_Size = 0;
#endif
#endif /* ASA_QUEUE */
#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_params */
  /* If not using RECUR_OPTIONS_FILE_DATA or data read from recur_asa_opt,
     store the recur_parameter ranges
     store the recur_parameter types
     store the initial recur_parameter values
     other changes needed for initialization */
#endif /* MY_TEMPLATE recur_params */
  RECUR_USER_OPTIONS->Asa_Recursive_Level = 1;

  return (0);
}

/***********************************************************************
* double recur_cost_function
*	This is the users cost function to optimize
*	(find the minimum).
*	cost_flag is set to TRUE if the parameter set
*	does not violates any constraints
*       recur_parameter_lower_bound and recur_parameter_upper_bound
*       may be adaptively changed during the search.
***********************************************************************/
#if HAVE_ANSI
double
recur_cost_function (double *x,
                     double *recur_parameter_lower_bound,
                     double *recur_parameter_upper_bound,
                     double *recur_cost_tangents,
                     double *recur_cost_curvature,
                     ALLOC_INT * recur_parameter_dimension,
                     int *recur_parameter_int_real,
                     int *recur_cost_flag,
                     int *recur_exit_code, USER_DEFINES * RECUR_USER_OPTIONS)
#else
double
recur_cost_function (x,
                     recur_parameter_lower_bound,
                     recur_parameter_upper_bound,
                     recur_cost_tangents,
                     recur_cost_curvature,
                     recur_parameter_dimension,
                     recur_parameter_int_real,
                     recur_cost_flag, recur_exit_code, RECUR_USER_OPTIONS)
     double *x;
     double *recur_parameter_lower_bound;
     double *recur_parameter_upper_bound;
     double *recur_cost_tangents;
     double *recur_cost_curvature;
     ALLOC_INT *recur_parameter_dimension;
     int *recur_parameter_int_real;
     int *recur_cost_flag;
     int *recur_exit_code;
     USER_DEFINES *RECUR_USER_OPTIONS;
#endif
{
  int fscanf_ret;
  double cost_value;
  static LONG_INT recur_funevals = 0;
  int *exit_code;
  char user_exit_msg[160];
#if OPTIONAL_DATA_PTR
  int data_ptr_flg;
#endif
#if OPTIONS_FILE
  FILE *ptr_options;
  char read_option[80];
  char read_if[4], read_FALSE[6], read_comm1[3], read_ASA_SAVE[9],
    read_comm2[3];
  int read_int;
#if INT_LONG
  LONG_INT read_long;
#endif
  double read_double;
#endif
#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_cost_decl */
  /* add some declarations if required */
#endif

  double *parameter_lower_bound, *parameter_upper_bound;
  double *cost_parameters;
  double *cost_tangents, *cost_curvature;
  ALLOC_INT *parameter_dimension;
  int *parameter_int_real;
  int *cost_flag;
  static LONG_INT *rand_seed;
  static int initial_flag = 0;
#if MULTI_MIN
  int multi_index;
#endif

  USER_DEFINES *USER_OPTIONS;

  recur_funevals = recur_funevals + 1;

  if ((rand_seed = (ALLOC_INT *) calloc (1, sizeof (ALLOC_INT))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): rand_seed");
    Exit_USER (user_exit_msg);
    return (-2);
  }

  if ((USER_OPTIONS =
       (USER_DEFINES *) calloc (1, sizeof (USER_DEFINES))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): USER_OPTIONS");
    Exit_USER (user_exit_msg);
    return (-2);
  }
#if OPTIONS_FILE
  /* Test to see if asa_opt is in correct directory.
     This is useful for some PC and Mac compilers. */
  if ((ptr_options = fopen ("asa_opt", "r")) == NULL) {
#if INCL_STDOUT
    printf ("\n\n*** EXIT fopen asa_opt failed *** \n\n");
#endif /* INCL_STDOUT */
    free (USER_OPTIONS);
    return (6);
  }

  fscanf_ret = fscanf (ptr_options, "%s%s%s%s%s",
                       read_if, read_FALSE, read_comm1, read_ASA_SAVE,
                       read_comm2);
  if (strcmp (read_if, "#if") || strcmp (read_FALSE, "FALSE")
      || strcmp (read_comm1, "/*") || strcmp (read_ASA_SAVE, "ASA_SAVE")
      || strcmp (read_comm2, "*/")) {
#if INCL_STDOUT
    printf ("\n\n*** EXIT not asa_opt for this version *** \n\n");
#endif /* INCL_STDOUT */
    fclose (ptr_options);
    free (USER_OPTIONS);
    return (-6);
  }
#if INT_LONG
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%ld", &read_long);
  USER_OPTIONS->Limit_Acceptances = read_long;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%ld", &read_long);
  USER_OPTIONS->Limit_Generated = read_long;
#else
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Limit_Acceptances = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Limit_Generated = read_int;
#endif
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Limit_Invalid_Generated_States = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Accepted_To_Generated_Ratio = read_double;

  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Cost_Precision = read_double;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Maximum_Cost_Repeat = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Number_Cost_Samples = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Temperature_Ratio_Scale = read_double;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Cost_Parameter_Scale_Ratio = read_double;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Temperature_Anneal_Scale = read_double;

  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Include_Integer_Parameters = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->User_Initial_Parameters = read_int;
#if INT_ALLOC
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Sequential_Parameters = read_int;
#else
#if INT_LONG
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%ld", &read_long);
  USER_OPTIONS->Sequential_Parameters = read_long;
#else
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Sequential_Parameters = read_int;
#endif
#endif
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Initial_Parameter_Temperature = read_double;

  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Acceptance_Frequency_Modulus = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Generated_Frequency_Modulus = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Reanneal_Cost = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Reanneal_Parameters = read_int;

  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%lf", &read_double);
  USER_OPTIONS->Delta_X = read_double;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->User_Tangents = read_int;
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  USER_OPTIONS->Curvature_0 = read_int;
#else /* OPTIONS_FILE */
  /* USER_OPTIONS->Limit_Acceptances = 10000; */
  USER_OPTIONS->Limit_Acceptances = 1000;
  USER_OPTIONS->Limit_Generated = 99999;
  USER_OPTIONS->Limit_Invalid_Generated_States = 1000;
  USER_OPTIONS->Accepted_To_Generated_Ratio = 1.0E-6;

  USER_OPTIONS->Cost_Precision = 1.0E-18;
  USER_OPTIONS->Maximum_Cost_Repeat = 2;
  USER_OPTIONS->Number_Cost_Samples = 2;

  /* These variables are set below in x[.] */
  /* USER_OPTIONS->Temperature_Ratio_Scale = 1.0E-5; */
  /* USER_OPTIONS->Cost_Parameter_Scale_Ratio = 1.0; */

  USER_OPTIONS->Temperature_Anneal_Scale = 100.;

  USER_OPTIONS->Include_Integer_Parameters = FALSE;
  USER_OPTIONS->User_Initial_Parameters = FALSE;
  USER_OPTIONS->Sequential_Parameters = -1;
  USER_OPTIONS->Initial_Parameter_Temperature = 1.0;

  USER_OPTIONS->Acceptance_Frequency_Modulus = 100;
  USER_OPTIONS->Generated_Frequency_Modulus = 10000;
  USER_OPTIONS->Reanneal_Cost = 1;
  USER_OPTIONS->Reanneal_Parameters = TRUE;

  USER_OPTIONS->Delta_X = 0.001;
  USER_OPTIONS->User_Tangents = FALSE;
  USER_OPTIONS->Curvature_0 = TRUE;
#endif /* OPTIONS_FILE */

  USER_OPTIONS->Temperature_Ratio_Scale = x[0];
  USER_OPTIONS->Cost_Parameter_Scale_Ratio = x[1];

  if (initial_flag == 0) {
    /* first value of *rand_seed */
#if ASA_LIB
    *rand_seed = (asa_rand_seed ? *asa_rand_seed : (LONG_INT) 696969);
#else
    *rand_seed = 696969;
#endif
  }

  if ((parameter_dimension =
       (ALLOC_INT *) calloc (1, sizeof (ALLOC_INT))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): parameter_dimension");
    Exit_USER (user_exit_msg);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  if ((exit_code = (int *) calloc (1, sizeof (int))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): exit_code");
    Exit_USER (user_exit_msg);
    free (parameter_dimension);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  if ((cost_flag = (int *) calloc (1, sizeof (int))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): cost_flag");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }

  /* the number of parameters for the cost function */
#if OPTIONS_FILE_DATA
  fscanf_ret = fscanf (ptr_options, "%s", read_option);
  fscanf_ret = fscanf (ptr_options, "%s", read_option);

#if INT_ALLOC
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  *parameter_dimension = read_int;
#else
#if INT_LONG
  fscanf_ret = fscanf (ptr_options, "%ld", &read_long);
  *parameter_dimension = read_long;
#else
  fscanf_ret = fscanf (ptr_options, "%d", &read_int);
  *parameter_dimension = read_int;
#endif
#endif

#else /* OPTIONS_FILE_DATA */
#if ASA_TEST
  /* set parameter dimension if SELF_OPTIMIZE=TRUE */
  *parameter_dimension = 4;
#endif /* ASA_TEST */
#endif /* OPTIONS_FILE_DATA */
#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_dim */
  /* If not using OPTIONS_FILE_DATA or data read from asa_opt,
     set parameter dimension if SELF_OPTIMIZE=TRUE */
#endif /* MY_TEMPLATE recur_dim */

  if ((parameter_lower_bound =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): parameter_lower_bound");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (cost_flag);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  if ((parameter_upper_bound =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): parameter_upper_bound");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (cost_flag);
    free (parameter_lower_bound);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  if ((cost_parameters =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): cost_parameters");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (cost_flag);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  if ((parameter_int_real =
       (int *) calloc (*parameter_dimension, sizeof (int))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): parameter_int_real");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (cost_flag);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (cost_parameters);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  if ((cost_tangents =
       (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "recur_cost_function(): cost_tangents");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (cost_flag);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (cost_parameters);
    free (parameter_int_real);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }

  if (USER_OPTIONS->Curvature_0 == FALSE || USER_OPTIONS->Curvature_0 == -1) {
    if ((cost_curvature =
         (double *) calloc ((*parameter_dimension) *
                            (*parameter_dimension),
                            sizeof (double))) == NULL) {
      strcpy (user_exit_msg, "recur_cost_function(): cost_curvature");
      Exit_USER (user_exit_msg);
      free (exit_code);
      free (parameter_dimension);
      free (cost_flag);
      free (parameter_lower_bound);
      free (parameter_upper_bound);
      free (cost_parameters);
      free (parameter_int_real);
      free (cost_tangents);
      free (USER_OPTIONS);
      fclose (ptr_options);
      return (-2);
    }
  } else {
    cost_curvature = (double *) NULL;
  }

#if ASA_TEMPLATE_SELFOPT
  /* Set memory to that required for use. */
  USER_OPTIONS->Asa_Data_Dim_Dbl = 2;
  if ((USER_OPTIONS->Asa_Data_Dbl =
       (double *) calloc (USER_OPTIONS->Asa_Data_Dim_Dbl,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg,
            "recur_cost_function(): USER_OPTIONS->Asa_Data_Dbl");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (cost_flag);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (cost_parameters);
    free (parameter_int_real);
    free (cost_tangents);
    free (cost_curvature);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
  /* Use Asa_Data_Dbl[0] as flag, e.g., if used with SELF_OPTIMIZE. */
  USER_OPTIONS->Asa_Data_Dbl[0] = 1.0;
#endif /* ASA_TEMPLATE_SELFOPT */

#if USER_COST_SCHEDULE
  USER_OPTIONS->Cost_Schedule = user_cost_schedule;
#endif
#if USER_ACCEPTANCE_TEST
  USER_OPTIONS->Acceptance_Test = user_acceptance_test;
#endif
#if USER_ACCEPT_ASYMP_EXP
  USER_OPTIONS->Asymp_Exp_Param = 1.0;
#endif
#if USER_GENERATING_FUNCTION
  USER_OPTIONS->Generating_Distrib = user_generating_distrib;
#endif
#if USER_REANNEAL_COST
  USER_OPTIONS->Reanneal_Cost_Function = user_reanneal_cost;
#endif
#if USER_REANNEAL_PARAMETERS
  USER_OPTIONS->Reanneal_Params_Function = user_reanneal_params;
#endif

  initialize_parameters (cost_parameters,
                         parameter_lower_bound,
                         parameter_upper_bound,
                         cost_tangents,
                         cost_curvature,
                         parameter_dimension, parameter_int_real,
#if OPTIONS_FILE_DATA
                         ptr_options,
#endif
                         USER_OPTIONS);
#if OPTIONS_FILE
  fclose (ptr_options);
#endif

#if ASA_SAVE
  USER_OPTIONS->Random_Array_Dim = SHUFFLE;
  USER_OPTIONS->Random_Array = random_array;
#endif /* ASA_SAVE */

  /* It might be a good idea to place a loop around this call,
     and to average over several values of funevals returned by
     trajectories of cost_value. */

  funevals = 0;

#if USER_ASA_USR_OUT
  if ((USER_OPTIONS->Asa_Usr_Out_File =
       (char *) calloc (80, sizeof (char))) == NULL) {
    strcpy (user_exit_msg,
            "recur_cost_function(): USER_OPTIONS->Asa_Usr_Out_File");
  }
#endif
#if USER_ASA_OUT
  if ((USER_OPTIONS->Asa_Out_File =
       (char *) calloc (80, sizeof (char))) == NULL) {
    strcpy (user_exit_msg,
            "recur_cost_function(): USER_OPTIONS->Asa_Out_File");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (cost_flag);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (cost_parameters);
    free (parameter_int_real);
    free (cost_tangents);
    free (cost_curvature);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
#if ASA_TEMPLATE_SELFOPT
  strcpy (USER_OPTIONS->Asa_Out_File, "asa_rcur");
#endif
#endif

#if OPTIONAL_DATA_PTR
  data_ptr_flg = 1;
#if ASA_TEMPLATE
  /* N.b.:  If OPTIONAL_DATA_PTR is being used for RECUR_USER_OPTIONS
   * as well as for USER_OPTIONS, do not create (or free) additional memory
   * in recur_cost_function() for Asa_Data_Dim_Ptr and Asa_Data_Ptr to
   * be passed to the inner cost_function(), but rather link pointers to
   * those in RECUR_USER_OPTIONS.  Typically, define separate structures
   * within the structure defined by Asa_Data_Ptr to access info depending
   * on whether the run in a particular level of cost function in this
   * recursive operation.  In this case, set * #if TRUE to #if FALSE just
   * below.  See the ASA-README for more discussion.
   */

#if TRUE
  USER_OPTIONS->Asa_Data_Dim_Ptr = 1;
  if ((USER_OPTIONS->Asa_Data_Ptr =
       (OPTIONAL_PTR_TYPE *) calloc (USER_OPTIONS->Asa_Data_Dim_Ptr,
                                     sizeof (OPTIONAL_PTR_TYPE))) == NULL) {
    strcpy (user_exit_msg,
            "recur_cost_function(): USER_OPTIONS->Asa_Data_Ptr");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (cost_flag);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (cost_parameters);
    free (parameter_int_real);
    free (cost_tangents);
    free (cost_curvature);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
#else
  USER_OPTIONS->Asa_Data_Dim_Ptr = RECUR_USER_OPTIONS->Asa_Data_Dim_Ptr;
  USER_OPTIONS->Asa_Data_Ptr = RECUR_USER_OPTIONS->Asa_Data_Ptr;
  data_ptr_flg = 0;
#endif
#endif /* ASA_TEMPLATE */
  /* see note at "Instead of freeing Asa_Data_Ptr" */
  /* USER_OPTIONS->Asa_Data_Ptr = OptionalPointer; */
  USER_OPTIONS->Asa_Data_Dim_Ptr = 1;
  if ((USER_OPTIONS->Asa_Data_Ptr =
       (OPTIONAL_PTR_TYPE *) calloc (USER_OPTIONS->Asa_Data_Dim_Ptr,
                                     sizeof (OPTIONAL_PTR_TYPE))) == NULL) {
    strcpy (user_exit_msg,
            "recur_cost_function(): USER_OPTIONS->Asa_Data_Ptr");
    Exit_USER (user_exit_msg);
    free (exit_code);
    free (parameter_dimension);
    free (cost_flag);
    free (parameter_lower_bound);
    free (parameter_upper_bound);
    free (cost_parameters);
    free (parameter_int_real);
    free (cost_tangents);
    free (cost_curvature);
    free (USER_OPTIONS);
    fclose (ptr_options);
    return (-2);
  }
#endif /* OPTIONAL_DATA_PTR */

  cost_value = asa (USER_COST_FUNCTION,
                    randflt,
                    rand_seed,
                    cost_parameters,
                    parameter_lower_bound,
                    parameter_upper_bound,
                    cost_tangents,
                    cost_curvature,
                    parameter_dimension,
                    parameter_int_real, cost_flag, exit_code, USER_OPTIONS);
  if (*exit_code == -1) {
#if INCL_STDOUT
    printf ("\n\n*** error in calloc in ASA ***\n\n");
#endif /* INCL_STDOUT */
    return (-1);
  }
#if MY_TEMPLATE                 /* MY_TEMPLATE_recur_post_asa */
#endif

  if (cost_value > .001) {
    *recur_cost_flag = FALSE;
  } else {
    *recur_cost_flag = TRUE;
  }

#if FALSE                       /* set to 1 to activate FAST EXIT */
  /* Make a quick exit */
  if (recur_funevals >= 10) {
    *recur_cost_flag = FALSE;
    RECUR_USER_OPTIONS->Limit_Invalid_Generated_States = 0;
#if INCL_STDOUT
    printf ("FAST EXIT set at recur_funevals = 10\n\n");
#endif
  }
#endif

#if TIME_CALC
  /* print every RECUR_PRINT_FREQUENCY evaluations */
  if ((RECUR_PRINT_FREQUENCY > 0) &&
      ((recur_funevals % RECUR_PRINT_FREQUENCY) == 0)) {
    USER_OPTIONS->Temperature_Ratio_Scale = x[0];
    USER_OPTIONS->Cost_Parameter_Scale_Ratio = x[1];
#if INCL_STDOUT
    printf ("USER_OPTIONS->Temperature_Ratio_Scale = %12.7g\n",
            USER_OPTIONS->Temperature_Ratio_Scale);
    printf ("USER_OPTIONS->Cost_Parameter_Scale_Ratio = %12.7g\n",
            USER_OPTIONS->Cost_Parameter_Scale_Ratio);
#endif
  }
#endif

#if INCL_STDOUT
  printf ("recur_funevals = %ld, *recur_cost_flag = %d\n",
          recur_funevals, *recur_cost_flag);
#endif
  /* cost function = number generated at best cost */
#if ASA_TEMPLATE_SELFOPT
  funevals = (LONG_INT) (USER_OPTIONS->Asa_Data_Dbl[1]);
#if INCL_STDOUT
  printf ("\tbest_funevals = %ld, cost_value = %12.7g\n\n",
          funevals, cost_value);
#endif
  /* cost function = total number generated during run */
#endif /* ASA_TEMPLATE_SELFOPT */

#if OPTIONAL_DATA_DBL
  free (USER_OPTIONS->Asa_Data_Dbl);
#endif
#if OPTIONAL_DATA_INT
  free (USER_OPTIONS->Asa_Data_Int);
#endif
#if OPTIONAL_DATA_PTR
  if (data_ptr_flg == 1) {
    free (USER_OPTIONS->Asa_Data_Ptr);
  }
#endif
#if USER_ASA_OUT
#if TEMPLATE
  /* if necessary */
  free (asa_out_my);
#endif
  free (USER_OPTIONS->Asa_Out_File);
#endif
#if USER_ASA_USR_OUT
#if ASA_TEMPLATE
  /* if necessary */
  free (asa_usr_out_my);
#endif
  free (USER_OPTIONS->Asa_Usr_Out_File);
#endif
#if ASA_QUEUE
#if ASA_RESOLUTION
#else
  free (USER_OPTIONS->Queue_Resolution);
#endif
#endif
#if ASA_RESOLUTION
  free (USER_OPTIONS->Coarse_Resolution);
#endif
  if (USER_OPTIONS->Curvature_0 == FALSE || USER_OPTIONS->Curvature_0 == -1)
    free (cost_curvature);
#if USER_INITIAL_PARAMETERS_TEMPS
  free (USER_OPTIONS->User_Parameter_Temperature);
#endif
#if USER_INITIAL_COST_TEMP
  free (USER_OPTIONS->User_Cost_Temperature);
#endif
#if DELTA_PARAMETERS
  free (USER_OPTIONS->User_Delta_Parameter);
#endif
#if QUENCH_PARAMETERS
  free (USER_OPTIONS->User_Quench_Param_Scale);
#endif
#if QUENCH_COST
  free (USER_OPTIONS->User_Quench_Cost_Scale);
#endif
#if RATIO_TEMPERATURE_SCALES
  free (USER_OPTIONS->User_Temperature_Ratio);
#endif
#if MULTI_MIN
  free (USER_OPTIONS->Multi_Grid);
  for (multi_index = 0; multi_index < USER_OPTIONS->Multi_Number;
       ++multi_index) {
    free (USER_OPTIONS->Multi_Params[multi_index]);
  }
#endif /* MULTI_MIN */
#if OPTIONAL_DATA_PTR
  if (data_ptr_flg == 0) {
    USER_OPTIONS = NULL;
  }
#endif
  free (USER_OPTIONS);
  free (parameter_dimension);
  free (exit_code);
  free (cost_flag);
  free (parameter_lower_bound);
  free (parameter_upper_bound);
  free (cost_parameters);
  free (parameter_int_real);
  free (cost_tangents);
  free (rand_seed);

  return ((double) funevals);
}

#if USER_COST_SCHEDULE
#if HAVE_ANSI
double
recur_user_cost_schedule (double test_temperature, const void *OPTIONS_TMP)
#else
double
recur_user_cost_schedule (test_temperature, OPTIONS_TMP)
     double test_temperature;
     void *OPTIONS_TMP;
#endif /* HAVE_ANSI */
{
  double x;
  USER_DEFINES *RECUR_USER_OPTIONS;

  RECUR_USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

  x = 0;                        /* initialize  to prevent warning */
#if ASA_TEMPLATE
  x = test_temperature;

#endif

  return (x);
}
#endif /* USER_COST_SCHEDULE */

#if USER_ACCEPTANCE_TEST
#if HAVE_ANSI
void
recur_user_acceptance_test (double current_cost,
                            double *recur_parameter_lower_bound,
                            double *recur_parameter_upper_bound,
                            ALLOC_INT * recur_parameter_dimension,
                            const void *OPTIONS_TMP)
#else
void
recur_user_acceptance_test (current_cost, recur_parameter_lower_bound,
                            recur_parameter_upper_bound,
                            recur_parameter_dimension, OPTIONS_TMP)
     double current_cost;
     double *recur_parameter_lower_bound;
     double *recur_parameter_upper_bound;
     ALLOC_INT *recur_parameter_dimension;
     void *OPTIONS_TMP;
#endif /* HAVE_ANSI */
{
  double uniform_test, curr_cost_temp;
#if USER_ACCEPT_ASYMP_EXP
  double x, q, delta_cost;
#endif
  USER_DEFINES *RECUR_USER_OPTIONS;

  RECUR_USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

#if ASA_TEMPLATE                /* ASA cost index */
  /* Calculate the current ASA cost index.  This could be useful
     to define a new schedule for the cost temperature, beyond
     simple changes that can be made using USER_COST_SCHEDULE. */

  int index;
  double k_temperature, quench, y;
  double xrecur_parameter_dimension;

#if QUENCH_COST
  quench = RECUR_USER_OPTIONS->User_Quench_Cost_Scale[0];
#else
  quench = 1.0;
#endif /* QUENCH_COST */
  xrecur_parameter_dimension = (double) *recur_parameter_dimension;
  for (index = 0; index < *recur_parameter_dimension; ++index)
    if (fabs
        (recur_parameter_upper_bound[index] -
         recur_parameter_lower_bound[index]) < (double) EPS_DOUBLE)
      *xrecur_parameter_dimension -= 1.0;

  y = -F_LOG (RECUR_USER_OPTIONS->Cost_Temp_Curr
              / RECUR_USER_OPTIONS->Cost_Temp_Init)
    / RECUR_USER_OPTIONS->Cost_Temp_Scale;

  k_temperature = F_POW (y, xrecur_parameter_dimension / quench);
#endif /* ASA cost index */

  uniform_test = randflt (RECUR_USER_OPTIONS->Random_Seed);
  curr_cost_temp = RECUR_USER_OPTIONS->Cost_Temp_Curr;

#if ASA_TEMPLATE
#if USER_COST_SCHEDULE
  curr_cost_temp =
    (RECUR_USER_OPTIONS->Cost_Schedule (RECUR_USER_OPTIONS->Cost_Temp_Curr,
                                        RECUR_USER_OPTIONS)
     + (double) EPS_DOUBLE);
#else
  curr_cost_temp = RECUR_USER_OPTIONS->Cost_Temp_Curr;
#endif
#endif /* ASA_TEMPLATE */

#if USER_ACCEPT_ASYMP_EXP
#if USER_COST_SCHEDULE
  curr_cost_temp =
    (RECUR_USER_OPTIONS->Cost_Schedule (RECUR_USER_OPTIONS->Cost_Temp_Curr,
                                        RECUR_USER_OPTIONS)
     + (double) EPS_DOUBLE);
#endif

  delta_cost = (current_cost - *(RECUR_USER_OPTIONS->Last_Cost))
    / (curr_cost_temp + (double) EPS_DOUBLE);

  q = RECUR_USER_OPTIONS->Asymp_Exp_Param;
  if (fabs (1.0 - q) < (double) EPS_DOUBLE)
    x = MIN (1.0, (F_EXP (-delta_cost)));       /* Boltzmann test */
  else if ((1.0 - (1.0 - q) * delta_cost) < (double) EPS_DOUBLE)
    x = MIN (1.0, (F_EXP (-delta_cost)));       /* Boltzmann test */
  else
    x = MIN (1.0, F_POW ((1.0 - (1.0 - q) * delta_cost), (1.0 / (1.0 - q))));

  RECUR_USER_OPTIONS->Prob_Bias = x;
  if (x >= uniform_test)
    RECUR_USER_OPTIONS->User_Acceptance_Flag = TRUE;
  else
    RECUR_USER_OPTIONS->User_Acceptance_Flag = FALSE;

#endif /* USER_ACCEPT_ASYMP_EXP */
}
#endif /* USER_ACCEPTANCE_TEST */

#if USER_GENERATING_FUNCTION
#if HAVE_ANSI
double
recur_user_generating_distrib (LONG_INT * seed,
                               ALLOC_INT * recur_parameter_dimension,
                               ALLOC_INT index_v,
                               double temperature_v,
                               double init_param_temp_v,
                               double temp_scale_params_v,
                               double parameter_v,
                               double parameter_range_v,
                               double *last_saved_parameter,
                               const void *OPTIONS_TMP)
#else
double
recur_user_generating_distrib (seed,
                               recur_parameter_dimension,
                               index_v,
                               temperature_v,
                               init_param_temp_v,
                               temp_scale_params_v,
                               parameter_v,
                               parameter_range_v,
                               last_saved_parameter, OPTIONS_TMP)
     LONG_INT *seed;
     ALLOC_INT *recur_parameter_dimension;
     ALLOC_INT index_v;
     double temperature_v;
     double init_param_temp_v;
     double temp_scale_params_v;
     double parameter_v;
     double parameter_range_v;
     double *last_saved_parameter;
     void *OPTIONS_TMP;
#endif
{
  double x;
  USER_DEFINES *RECUR_USER_OPTIONS;

  RECUR_USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

  x = 0;                        /* initialize  to prevent warning */
#if ASA_TEMPLATE
  double x, y, z;

  /* This is the ASA distribution.  A slower temperature schedule can be
     obtained here, e.g., temperature_v = pow(temperature_v, 0.5); */

  x = randflt (seed);
  y = x < 0.5 ? -1.0 : 1.0;
  z = y * temperature_v * (F_POW ((1.0 + 1.0 / temperature_v),
                                  fabs (2.0 * x - 1.0)) - 1.0);

  x = parameter_v + z * parameter_range_v;
#endif /* ASA_TEMPLATE */

  return (x);
}
#endif /* USER_GENERATING_FUNCTION */

#if USER_REANNEAL_COST
#if HAVE_ANSI
int
recur_user_reanneal_cost (double *cost_best,
                          double *cost_last,
                          double *initial_cost_temperature,
                          double *current_cost_temperature,
                          const void *OPTIONS_TMP)
#else
int
recur_user_reanneal_cost (cost_best,
                          cost_last,
                          initial_cost_temperature,
                          current_cost_temperature, OPTIONS_TMP)
     double *cost_best;
     double *cost_last;
     double *initial_cost_temperature;
     double *current_cost_temperature;
     void *OPTIONS_TMP;
#endif /* HAVE_ANSI */
{
  USER_DEFINES *RECUR_USER_OPTIONS;

  RECUR_USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

#if ASA_TEMPLATE
  double tmp_dbl;

  tmp_dbl = MAX (fabs (*cost_last), fabs (*cost_best));
  tmp_dbl = MAX ((double) EPS_DOUBLE, tmp_dbl);
  *initial_cost_temperature = MIN (*initial_cost_temperature, tmp_dbl);

#endif

  return (TRUE);
}
#endif /* USER_REANNEAL_COST */

#if USER_REANNEAL_PARAMETERS
#if HAVE_ANSI
double
recur_user_reanneal_params (double current_temp,
                            double tangent,
                            double max_tangent, const void *OPTIONS_TMP)
#else
double
recur_user_reanneal_params (current_temp, tangent, max_tangent, OPTIONS_TMP)
     double current_temp;
     double tangent;
     double max_tangent;
     void *OPTIONS_TMP;
#endif /* HAVE_ANSI */
{
  double x;
  USER_DEFINES *RECUR_USER_OPTIONS;

  RECUR_USER_OPTIONS = (USER_DEFINES *) OPTIONS_TMP;

  x = 0;                        /* initialize  to prevent warning */

#if ASA_TEMPLATE
  x = current_temp * (max_tangent / tangent);
#endif

  return (x);
}
#endif /* USER_REANNEAL_PARAMETERS */
#endif /* SELF_OPTIMIZE */

#if FITLOC
#if HAVE_ANSI
double
calcf (double (*user_cost_function)

        
       (double *, double *, double *, double *, double *, ALLOC_INT *, int *,
        int *, int *, USER_DEFINES *), double *xloc,
       double *parameter_lower_bound, double *parameter_upper_bound,
       double *cost_tangents, double *cost_curvature,
       ALLOC_INT * parameter_dimension, int *parameter_int_real,
       int *cost_flag, int *exit_code, USER_DEFINES * OPTIONS, FILE * ptr_out)
#else
double
calcf (user_cost_function,
       xloc,
       parameter_lower_bound,
       parameter_upper_bound,
       cost_tangents,
       cost_curvature,
       parameter_dimension,
       parameter_int_real, cost_flag, exit_code, OPTIONS, ptr_out)
     double (*user_cost_function) ();
     double *x;
     double *parameter_lower_bound;
     double *parameter_upper_bound;
     double *cost_tangents;
     double *cost_curvature;
     ALLOC_INT *parameter_dimension;
     int *parameter_int_real;
     int *cost_flag;
     int *exit_code;
     USER_DEFINES *OPTIONS;
     FILE *ptr_out;
#endif
{
  ALLOC_INT index_v;
#if FITLOC_ROUND
  double x, min_parameter_v, max_parameter_v, parameter_range_v;
#if ASA_RESOLUTION
  double xres, xint, xplus, xminus, dx, dxminus, dxplus;
#endif
#endif
  double floc;

#if FITLOC_ROUND
  /* The following section for adjustments of parameters is taken from
     generate_new_state() in asa.c */
  for (index_v = 0; index_v < *parameter_dimension; ++index_v) {
    if (fabs
        (parameter_lower_bound[index_v] - parameter_upper_bound[index_v]) <
        EPS_DOUBLE)
      continue;

    x = xloc[index_v];

    min_parameter_v = parameter_lower_bound[index_v];
    max_parameter_v = parameter_upper_bound[index_v];
    parameter_range_v = max_parameter_v - min_parameter_v;

    /* Handle discrete parameters. */
#if ASA_RESOLUTION
    xres = OPTIONS->Coarse_Resolution[index_v];
    if (xres > EPS_DOUBLE) {
      min_parameter_v -= (xres / 2.0);
      max_parameter_v += (xres / 2.0);
      parameter_range_v = max_parameter_v - min_parameter_v;
    }
#endif /* ASA_RESOLUTION */
    if (parameter_int_real[index_v] > 0) {
#if ASA_RESOLUTION
      if (xres > EPS_DOUBLE) {
        ;
      } else {
#endif /* ASA_RESOLUTION */
        min_parameter_v -= 0.5;
        max_parameter_v += 0.5;
        parameter_range_v = max_parameter_v - min_parameter_v;
      }
#if ASA_RESOLUTION
    }
#endif
#if ASA_RESOLUTION
    if (xres > EPS_DOUBLE) {
      xint = xres * (double) ((LONG_INT) (x / xres));
      xplus = xint + xres;
      xminus = xint - xres;
      dx = fabs (xint - x);
      dxminus = fabs (xminus - x);
      dxplus = fabs (xplus - x);

      if (dx < dxminus && dx < dxplus)
        x = xint;
      else if (dxminus < dxplus)
        x = xminus;
      else
        x = xplus;
    }
#endif /* ASA_RESOLUTION */

    /* Handle discrete parameters.
       You might have to check rounding on your machine. */
    if (parameter_int_real[index_v] > 0) {
#if ASA_RESOLUTION
      if (xres > EPS_DOUBLE) {
        ;
      } else {
#endif /* ASA_RESOLUTION */
        if (x < min_parameter_v + 0.5)
          x = min_parameter_v + 0.5 + (double) EPS_DOUBLE;
        if (x > max_parameter_v - 0.5)
          x = max_parameter_v - 0.5 + (double) EPS_DOUBLE;

        if (x + 0.5 > 0.0) {
          x = (double) ((LONG_INT) (x + 0.5));
        } else {
          x = (double) ((LONG_INT) (x - 0.5));
        }
        if (x > parameter_upper_bound[index_v])
          x = parameter_upper_bound[index_v];
        if (x < parameter_lower_bound[index_v])
          x = parameter_lower_bound[index_v];
      }
#if ASA_RESOLUTION
    }
    if (xres > EPS_DOUBLE) {
      if (x < min_parameter_v + xres / 2.0)
        x = min_parameter_v + xres / 2.0 + (double) EPS_DOUBLE;
      if (x > max_parameter_v - xres / 2.0)
        x = max_parameter_v - xres / 2.0 + (double) EPS_DOUBLE;

      if (x > parameter_upper_bound[index_v])
        x = parameter_upper_bound[index_v];
      if (x < parameter_lower_bound[index_v])
        x = parameter_lower_bound[index_v];
    }
#endif /* ASA_RESOLUTION */
    if ((x < parameter_lower_bound[index_v])
        || (x > parameter_upper_bound[index_v])) {
      ;
    } else {
      xloc[index_v] = x;
    }
  }
#endif /* FITLOC_ROUND */

  floc = user_cost_function (xloc,
                             parameter_lower_bound,
                             parameter_upper_bound,
                             cost_tangents,
                             cost_curvature,
                             parameter_dimension,
                             parameter_int_real,
                             cost_flag, exit_code, OPTIONS);

  if (*cost_flag == FALSE) {
    floc += OPTIONS->Penalty;
  }

  for (index_v = 0; index_v < *parameter_dimension; ++index_v) {
    if (parameter_upper_bound[index_v] - xloc[index_v] < EPS_DOUBLE)
      floc += OPTIONS->Penalty;
    else if (xloc[index_v] - parameter_lower_bound[index_v] < EPS_DOUBLE)
      floc += OPTIONS->Penalty;
  }

  return (floc);
}

#if HAVE_ANSI
double
fitloc (double (*user_cost_function)

         
        (double *, double *, double *, double *, double *, ALLOC_INT *, int *,
         int *, int *, USER_DEFINES *), double *xloc,
        double *parameter_lower_bound, double *parameter_upper_bound,
        double *cost_tangents, double *cost_curvature,
        ALLOC_INT * parameter_dimension, int *parameter_int_real,
        int *cost_flag, int *exit_code, USER_DEFINES * OPTIONS,
        FILE * ptr_out)
#else
double
fitloc (user_cost_function,
        xloc,
        parameter_lower_bound,
        parameter_upper_bound,
        cost_tangents,
        cost_curvature,
        parameter_dimension,
        parameter_int_real, cost_flag, exit_code, OPTIONS, ptr_out)
     double (*user_cost_function) ();
     double *xloc;
     double *parameter_lower_bound;
     double *parameter_upper_bound;
     double *cost_tangents;
     double *cost_curvature;
     ALLOC_INT *parameter_dimension;
     int *parameter_int_real;
     int *cost_flag;
     int *exit_code;
     USER_DEFINES *OPTIONS;
     FILE *ptr_out;
#endif
{
  double x;
  ALLOC_INT index_v;
#if FITLOC_ROUND
  double min_parameter_v, max_parameter_v, parameter_range_v;
#if ASA_RESOLUTION
  double xres, xint, xminus, xplus, dx, dxminus, dxplus;
#endif
#endif
  double *xsave;
  double tol1, tol2, alpha, beta1, beta2, gamma, delta, floc, fsave, ffinal;
  int no_progress, tot_iters, locflg, bndflg;

#if FITLOC_PRINT
  if (OPTIONS->Fit_Local >= 1) {
    fprintf (ptr_out, "\n\nSTART LOCAL FIT\n");
  } else {
    fprintf (ptr_out, "\n\nSTART LOCAL FIT Independent of ASA\n");
  }
  fflush (ptr_out);
#endif /* FITLOC_PRINT */

  xsave = (double *) calloc (*parameter_dimension, sizeof (double));
  bndflg = 0;

  /* The following simplex parameters may need adjustments for your system. */
  tol1 = EPS_DOUBLE;
  tol2 = EPS_DOUBLE * 100.;
  no_progress = 4;
  alpha = 1.0;
  beta1 = 0.75;
  beta2 = 0.75;
  gamma = 1.25;
  delta = 2.50;

  for (index_v = 0; index_v < *parameter_dimension; ++index_v) {
    xsave[index_v] = xloc[index_v];
  }

  fsave = user_cost_function (xloc,
                              parameter_lower_bound,
                              parameter_upper_bound,
                              cost_tangents,
                              cost_curvature,
                              parameter_dimension,
                              parameter_int_real,
                              cost_flag, exit_code, OPTIONS);

  tot_iters = simplex (user_cost_function,
                       xloc,
                       parameter_lower_bound,
                       parameter_upper_bound,
                       cost_tangents,
                       cost_curvature,
                       parameter_dimension,
                       parameter_int_real,
                       cost_flag,
                       exit_code,
                       OPTIONS,
                       ptr_out,
                       tol1,
                       tol2, no_progress, alpha, beta1, beta2, gamma, delta);
  fflush (ptr_out);

  for (index_v = 0; index_v < *parameter_dimension; ++index_v) {
    x = xloc[index_v];
    if ((x < parameter_lower_bound[index_v])
        || (x > parameter_upper_bound[index_v])) {
      bndflg = 1;
    }
  }

  /* The following section for adjustments of parameters is taken from
     generate_new_state() in asa.c */
#if FITLOC_ROUND
  for (index_v = 0; index_v < *parameter_dimension; ++index_v) {
    if (fabs
        (parameter_lower_bound[index_v] - parameter_upper_bound[index_v]) <
        EPS_DOUBLE)
      continue;

    x = xloc[index_v];

    min_parameter_v = parameter_lower_bound[index_v];
    max_parameter_v = parameter_upper_bound[index_v];
    parameter_range_v = max_parameter_v - min_parameter_v;

    /* Handle discrete parameters. */
#if ASA_RESOLUTION
    xres = OPTIONS->Coarse_Resolution[index_v];
    if (xres > EPS_DOUBLE) {
      min_parameter_v -= (xres / 2.0);
      max_parameter_v += (xres / 2.0);
      parameter_range_v = max_parameter_v - min_parameter_v;
    }
#endif /* ASA_RESOLUTION */
    if (parameter_int_real[index_v] > 0) {
#if ASA_RESOLUTION
      if (xres > EPS_DOUBLE) {
        ;
      } else {
#endif /* ASA_RESOLUTION */
        min_parameter_v -= 0.5;
        max_parameter_v += 0.5;
        parameter_range_v = max_parameter_v - min_parameter_v;
      }
#if ASA_RESOLUTION
    }
#endif
#if ASA_RESOLUTION
    if (xres > EPS_DOUBLE) {
      xint = xres * (double) ((LONG_INT) (x / xres));
      xplus = xint + xres;
      xminus = xint - xres;
      dx = fabs (xint - x);
      dxminus = fabs (xminus - x);
      dxplus = fabs (xplus - x);

      if (dx < dxminus && dx < dxplus)
        x = xint;
      else if (dxminus < dxplus)
        x = xminus;
      else
        x = xplus;
    }
#endif /* ASA_RESOLUTION */

    /* Handle discrete parameters.
       You might have to check rounding on your machine. */
    if (parameter_int_real[index_v] > 0) {
#if ASA_RESOLUTION
      if (xres > EPS_DOUBLE) {
        ;
      } else {
#endif /* ASA_RESOLUTION */
        if (x < min_parameter_v + 0.5)
          x = min_parameter_v + 0.5 + (double) EPS_DOUBLE;
        if (x > max_parameter_v - 0.5)
          x = max_parameter_v - 0.5 + (double) EPS_DOUBLE;

        if (x + 0.5 > 0.0) {
          x = (double) ((LONG_INT) (x + 0.5));
        } else {
          x = (double) ((LONG_INT) (x - 0.5));
        }
        if (x > parameter_upper_bound[index_v])
          x = parameter_upper_bound[index_v];
        if (x < parameter_lower_bound[index_v])
          x = parameter_lower_bound[index_v];
      }
#if ASA_RESOLUTION
    }
    if (xres > EPS_DOUBLE) {
      if (x < min_parameter_v + xres / 2.0)
        x = min_parameter_v + xres / 2.0 + (double) EPS_DOUBLE;
      if (x > max_parameter_v - xres / 2.0)
        x = max_parameter_v - xres / 2.0 + (double) EPS_DOUBLE;

      if (x > parameter_upper_bound[index_v])
        x = parameter_upper_bound[index_v];
      if (x < parameter_lower_bound[index_v])
        x = parameter_lower_bound[index_v];
    }
#endif /* ASA_RESOLUTION */
    if ((x < parameter_lower_bound[index_v])
        || (x > parameter_upper_bound[index_v])) {
      bndflg = 1;
#if FITLOC_PRINT
      if (OPTIONS->Fit_Local == 2)
        fprintf (ptr_out, "IGNORE FITLOC: OUT OF BOUNDS xloc[%ld] = %g\n",
                 index_v, xloc[index_v]);
      else
        fprintf (ptr_out, "OUT OF BOUNDS xloc[%ld] = %g\n",
                 index_v, xloc[index_v]);
#else
      ;
#endif /* FITLOC_PRINT */
    } else {
      xloc[index_v] = x;
    }
  }
#endif /* FITLOC_ROUND */

  floc = user_cost_function (xloc,
                             parameter_lower_bound,
                             parameter_upper_bound,
                             cost_tangents,
                             cost_curvature,
                             parameter_dimension,
                             parameter_int_real,
                             cost_flag, exit_code, OPTIONS);

  if (fabs (floc - fsave) < (double) EPS_DOUBLE) {
    locflg = 1;
    ffinal = fsave;
#if FITLOC_PRINT
    fprintf (ptr_out, "\nsame global cost = %g\tlocal cost = %g\n\n",
             fsave, floc);
#endif /* FITLOC_PRINT */
  } else {
    if (floc < fsave) {
      if (OPTIONS->Fit_Local == 2 && bndflg == 1) {
        locflg = 1;
        ffinal = fsave;
      } else {
        locflg = 0;
        ffinal = floc;
      }
    } else {
      locflg = 1;
      ffinal = fsave;
    }
#if FITLOC_PRINT
    fprintf (ptr_out, "\nDIFF global cost = %g\tlocal cost = %g\n\n",
             fsave, floc);
#endif /* FITLOC_PRINT */
  }

  for (index_v = 0; index_v < *parameter_dimension; ++index_v) {
    if (fabs (xloc[index_v] - xsave[index_v]) < (double) EPS_DOUBLE) {
#if FITLOC_PRINT
      fprintf (ptr_out, "same global param[%ld] = %g\tlocal param = %g\n",
               index_v, xsave[index_v], xloc[index_v]);
#else
      ;
#endif /* FITLOC_PRINT */
    } else {
#if FITLOC_PRINT
      fprintf (ptr_out, "DIFF global param[%ld] = %g\tlocal param = %g\n",
               index_v, xsave[index_v], xloc[index_v]);
#else
      ;
#endif /* FITLOC_PRINT */
      if (locflg == 1) {
        xloc[index_v] = xsave[index_v];
      }
    }
  }

#if FITLOC_PRINT
  fprintf (ptr_out, "\n");
  fflush (ptr_out);
#endif /* FITLOC_PRINT */

  free (xsave);

  return (ffinal);
}

/*
   Written by Mark Johnson <mjohnson@netcom.com>, based on 

   %A J.A. Nelder
   %A R. Mead
   %T A simplex method for function minimization
   %J Computer J. (UK)
   %V 7
   %D 1964
   %P 308-313

   with improvements from

   %A G.P. Barabino
   %A G.S. Barabino
   %A B. Bianco
   %A M. Marchesi
   %T A study on the performances of simplex methods for function minimization
   %B Proc. IEEE Int. Conf. Circuits and Computers
   %D 1980
   %P 1150-1153

   adapted for use in ASA by Lester Ingber <ingber@ingber.com>
 */

#if HAVE_ANSI
int
simplex (double (*user_cost_function)

          
         (double *, double *, double *, double *, double *, ALLOC_INT *,
          int *, int *, int *, USER_DEFINES *), double *x,
         double *parameter_lower_bound, double *parameter_upper_bound,
         double *cost_tangents, double *cost_curvature,
         ALLOC_INT * parameter_dimension, int *parameter_int_real,
         int *cost_flag, int *exit_code, USER_DEFINES * OPTIONS,
         FILE * ptr_out, double tol1, double tol2, int no_progress,
         double alpha, double beta1, double beta2, double gamma, double delta)
#else
int
simplex (user_cost_function,
         x,
         parameter_lower_bound,
         parameter_upper_bound,
         cost_tangents,
         cost_curvature,
         parameter_dimension,
         parameter_int_real,
         cost_flag,
         exit_code,
         OPTIONS,
         ptr_out, tol1, tol2, no_progress, alpha, beta1, beta2, gamma, delta)
     double (*user_cost_function) ();
     double *x;
     double *parameter_lower_bound;
     double *parameter_upper_bound;
     double *cost_tangents;
     double *cost_curvature;
     ALLOC_INT *parameter_dimension;
     int *parameter_int_real;
     int *cost_flag;
     int *exit_code;
     USER_DEFINES *OPTIONS;
     FILE *ptr_out;
     double tol1;
     double tol2;
     int no_progress;
     double alpha;
     double beta1;
     double beta2;
     double gamma;
     double delta;
#endif
{
  double fs, fl, fh, fr, fe, fc1, fc2, ftmp, flast;
  double err1;
  double *fvals;
  double **splx;                /* the simplex of points */
  double *x0;                   /* centroid of simplex */
  double *xr;                   /* point for a reflection */
  double *xe;                   /* point for an expansion */
  double *xc1;                  /* point for a minor contraction */
  double *xc2;                  /* point for a major contraction */
  int s, l, h;
  int i, j, iters, futility;
  int lastprint;

  fvals = (double *) calloc (*parameter_dimension + 1, sizeof (double));
  splx = (double **) calloc (*parameter_dimension + 1, sizeof (double *));
  for (i = 0; i <= *parameter_dimension; i++)
    splx[i] = (double *) calloc (*parameter_dimension, sizeof (double));
  x0 = (double *) calloc (*parameter_dimension, sizeof (double));
  xr = (double *) calloc (*parameter_dimension, sizeof (double));
  xe = (double *) calloc (*parameter_dimension, sizeof (double));
  xc1 = (double *) calloc (*parameter_dimension, sizeof (double));
  xc2 = (double *) calloc (*parameter_dimension, sizeof (double));

  /* build the initial simplex */
  for (i = 0; i < *parameter_dimension; i++) {
    splx[0][i] = x[i];
  }
  for (i = 1; i <= *parameter_dimension; i++) {
    for (j = 0; j < *parameter_dimension; j++) {
      if ((j + 1) == i)
        splx[i][j] = (x[j] * 2.25) + tol2;
      else
        splx[i][j] = x[j];
      xr[j] = splx[i][j];
    }
    fvals[i] = calcf (user_cost_function,
                      xr,
                      parameter_lower_bound,
                      parameter_upper_bound,
                      cost_tangents,
                      cost_curvature,
                      parameter_dimension,
                      parameter_int_real,
                      cost_flag, exit_code, OPTIONS, ptr_out);
  }

  /* and of course compute function at starting point */
  fvals[0] = calcf (user_cost_function,
                    x,
                    parameter_lower_bound,
                    parameter_upper_bound,
                    cost_tangents,
                    cost_curvature,
                    parameter_dimension,
                    parameter_int_real,
                    cost_flag, exit_code, OPTIONS, ptr_out);

  /* now find the largest, 2nd largest, smallest f values */
  if (fvals[0] > fvals[1]) {
    h = 0;
    s = 1;
    l = 1;
  } else {
    h = 1;
    s = 0;
    l = 0;
  }
  fh = fvals[h];
  fs = fvals[s];
  fl = fvals[l];
  for (i = 2; i <= *parameter_dimension; i++) {
    if (fvals[i] <= fvals[l]) {
      l = i;
      fl = fvals[i];
    } else {
      if (fvals[i] >= fvals[h]) {
        s = h;
        fs = fh;
        h = i;
        fh = fvals[i];
      } else if (fvals[i] >= fvals[s]) {
        s = i;
        fs = fvals[i];
      }
    }
  }
#if FITLOC_PRINT
  if ((s == h) || (s == l) || (h == l))
    fprintf (ptr_out, "\nPANIC: s,l,h not unique %d %d %d\n", s, h, l);

  fprintf (ptr_out, "INITIAL SIMPLEX:\n");
  for (i = 0; i <= *parameter_dimension; i++) {
    for (j = 0; j < *parameter_dimension; j++) {
      fprintf (ptr_out, "   %11.4g", splx[i][j]);
    }
    fprintf (ptr_out, "      f = %12.5g", fvals[i]);
    if (i == h)
      fprintf (ptr_out, "  HIGHEST");
    if (i == s)
      fprintf (ptr_out, "  SECOND HIGHEST");
    if (i == l)
      fprintf (ptr_out, "  LOWEST");
    fprintf (ptr_out, "\n");
  }
#endif /* FITLOC_PRINT */

/* MAJOR LOOP */

  flast = fl;
  futility = 0;
  lastprint = 0;
  iters = 0;
  err1 = 1.1 + (1.1 * tol1);
  while ((err1 > tol1) && (iters < OPTIONS->Iter_Max) &&
         (futility < (*parameter_dimension * no_progress))) {
    iters++;
    /* now find the largest, 2nd largest, smallest f values */
    if (fvals[0] > fvals[1]) {
      h = 0;
      s = 1;
      l = 1;
    } else {
      h = 1;
      s = 0;
      l = 0;
    }
    fh = fvals[h];
    fs = fvals[s];
    fl = fvals[l];
    for (i = 2; i <= *parameter_dimension; i++) {
      if (fvals[i] <= fvals[l]) {
        l = i;
        fl = fvals[i];
      } else {
        if (fvals[i] >= fvals[h]) {
          s = h;
          fs = fh;
          h = i;
          fh = fvals[i];
        } else if (fvals[i] >= fvals[s]) {
          s = i;
          fs = fvals[i];
        }
      }
    }
#if FITLOC_PRINT
    if ((s == h) || (s == l) || (h == l))
      fprintf (ptr_out, "\nPANIC: s,l,h not unique %d %d %d\n", s, h, l);
#endif

    /* compute the centroid */
    for (j = 0; j < *parameter_dimension; j++) {
      x0[j] = 0.0;
      for (i = 0; i <= *parameter_dimension; i++) {
        if (i != h)
          x0[j] += splx[i][j];
      }
      x0[j] /= ((double) *parameter_dimension);
    }

    if (fl < flast) {
      flast = fl;
      futility = 0;
    } else
      futility += 1;

#if FITLOC_PRINT
    fprintf (ptr_out, "Iteration %3d f(best) = %12.6g halt? = %11.5g\n",
             iters, fl, err1);
    if ((iters - lastprint) >= 100) {
      fprintf (ptr_out, "\n     Best point seen so far:\n");
      for (i = 0; i < *parameter_dimension; i++) {
        fprintf (ptr_out, "     x[%3d] = %15.7g\n", i, splx[l][i]);
      }
      lastprint = iters;
      fprintf (ptr_out, "\n");
    }
    fflush (ptr_out);
#endif /* FITLOC_PRINT */

    /* STEP 1: compute a reflected point xr */
    for (i = 0; i < *parameter_dimension; i++) {
      xr[i] = ((1.0 + alpha) * x0[i]) - (alpha * splx[h][i]);
    }
    fr = calcf (user_cost_function,
                xr,
                parameter_lower_bound,
                parameter_upper_bound,
                cost_tangents,
                cost_curvature,
                parameter_dimension,
                parameter_int_real, cost_flag, exit_code, OPTIONS, ptr_out);

    /* typical: <2nd-biggest , >lowest .  Go again */
    if ((fr < fs) && (fr > fl)) {
      for (i = 0; i < *parameter_dimension; i++) {
        splx[h][i] = xr[i];
      }
      fvals[h] = fr;
      goto MORE_ITERS_asa_usr;
    }

    /* STEP 2: if reflected point is favorable, expand the simplex */
    if (fr < fl) {
      for (i = 0; i < *parameter_dimension; i++) {
        xe[i] = (gamma * xr[i]) + ((1.0 - gamma) * x0[i]);
      }
      fe = calcf (user_cost_function,
                  xe,
                  parameter_lower_bound,
                  parameter_upper_bound,
                  cost_tangents,
                  cost_curvature,
                  parameter_dimension,
                  parameter_int_real, cost_flag, exit_code, OPTIONS, ptr_out);
      if (fe < fr) {            /* win big; expansion point tiny */
        for (i = 0; i < *parameter_dimension; i++) {
          splx[h][i] = xe[i];
        }
        fvals[h] = fh = fe;
      } else
        /* still ok; reflection point a winner */
      {
        for (i = 0; i < *parameter_dimension; i++) {
          splx[h][i] = xr[i];
        }
        fvals[h] = fh = fr;
      }
      goto MORE_ITERS_asa_usr;
    }

    /* STEP 3: if reflected point is unfavorable, contract simplex */
    if (fr > fs) {
      if (fr < fh) {            /* may as well replace highest pt */
        for (i = 0; i < *parameter_dimension; i++) {
          splx[h][i] = xr[i];
        }
        fvals[h] = fh = fr;
      }
      for (i = 0; i < *parameter_dimension; i++) {
        xc1[i] = (beta1 * xr[i]) + ((1.0 - beta1) * x0[i]);
      }
      fc1 = calcf (user_cost_function,
                   xc1,
                   parameter_lower_bound,
                   parameter_upper_bound,
                   cost_tangents,
                   cost_curvature,
                   parameter_dimension,
                   parameter_int_real,
                   cost_flag, exit_code, OPTIONS, ptr_out);
      if (fc1 < fh) {           /* slight contraction worked */
        for (i = 0; i < *parameter_dimension; i++) {
          splx[h][i] = xc1[i];
        }
        fvals[h] = fh = fc1;
        goto MORE_ITERS_asa_usr;
      }
      /* now have to try strong contraction */
      for (i = 0; i < *parameter_dimension; i++) {
        xc2[i] = (beta2 * splx[h][i]) + ((1.0 - beta2) * x0[i]);
      }
      fc2 = calcf (user_cost_function,
                   xc2,
                   parameter_lower_bound,
                   parameter_upper_bound,
                   cost_tangents,
                   cost_curvature,
                   parameter_dimension,
                   parameter_int_real,
                   cost_flag, exit_code, OPTIONS, ptr_out);
      if (fc2 < fh) {           /* strong contraction worked */
        for (i = 0; i < *parameter_dimension; i++) {
          splx[h][i] = xc2[i];
        }
        fvals[h] = fh = fc2;
        goto MORE_ITERS_asa_usr;
      }
    }

    /* STEP 4: nothing worked.  collapse the simplex around xl */
    for (i = 0; i <= *parameter_dimension; i++) {
      if (i != l) {
        for (j = 0; j < *parameter_dimension; j++) {
          splx[i][j] = (splx[i][j] + splx[l][j]) / delta;
          xr[j] = splx[i][j];
        }
        fvals[i] = calcf (user_cost_function,
                          xr,
                          parameter_lower_bound,
                          parameter_upper_bound,
                          cost_tangents,
                          cost_curvature,
                          parameter_dimension,
                          parameter_int_real,
                          cost_flag, exit_code, OPTIONS, ptr_out);
      }
    }

  MORE_ITERS_asa_usr:

    ftmp = 0.00;
    for (i = 0; i <= *parameter_dimension; i++) {
      ftmp += fvals[i];
    }
    ftmp /= ((double) (*parameter_dimension + 1));

    err1 = 0.00;
    for (i = 0; i <= *parameter_dimension; i++) {
      err1 += ((fvals[i] - ftmp) * (fvals[i] - ftmp));
    }
    err1 /= ((double) (*parameter_dimension + 1));
    err1 = sqrt (err1);
  }                             /* end of major while loop */

  /* find the smallest f value */
  l = 0;
  fl = fvals[0];
  for (i = 1; i <= *parameter_dimension; i++) {
    if (fvals[i] < fvals[l])
      l = i;
  }

  /* give it back to the user */
  for (i = 0; i < *parameter_dimension; i++) {
    x[i] = splx[l][i];
  }

  free (fvals);
  for (i = 0; i <= *parameter_dimension; i++)
    free (splx[i]);
  free (splx);
  free (x0);
  free (xr);
  free (xe);
  free (xc1);
  free (xc2);

  return (iters);
}
#else
#endif /* FITLOC */

#if ASA_TEMPLATE_SAMPLE

#if HAVE_ANSI
void
sample (FILE * ptr_out, FILE * ptr_asa)
#else
void
sample (ptr_out, ptr_asa)
     FILE *ptr_out;
     FILE *ptr_asa;
#endif
{
  int fscanf_ret;
  int ind, n_samples, n_accept, index, dim;
  double cost, cost_temp, bias_accept;
  double param, temp, bias_gener, aver_weight, range;
  double sum, norm, answer, prod, binsize;
  char ch[80], sample[8];

  /*
     This is a demonstration of using ASA_SAMPLE to perform the double integral
     of exp(-x^2 - y^2) for x and y between 0 and 2.  The mesh is quite crude.

     The temperature-dependent acceptance and generated biases factor are
     divided out, and the actual cost function weights each point.
   */

  dim = 2;
  norm = sum = 0.;
  n_samples = 0;

  fprintf (ptr_out,
           ":SAMPLE:   n_accept   cost        cost_temp    bias_accept    aver_weight\n");
  fprintf (ptr_out,
           ":SAMPLE:   index      param[]     temp[]       bias_gener[]   range[]\n");
  for (;;) {
    fscanf_ret = fscanf (ptr_asa, "%s", ch);
    if (!strcmp (ch, "exit_status")) {
      break;
    }
    if (strcmp (ch, ":SAMPLE#")) {
      continue;
    }
    ++n_samples;
    fprintf (ptr_out, "%s\n", ch);
    fflush (ptr_out);
    fscanf_ret = fscanf (ptr_asa, "%s%d%lf%lf%lf%lf",
                         sample, &n_accept, &cost, &cost_temp, &bias_accept,
                         &aver_weight);
    if (strcmp (sample, ":SAMPLE+")) {
      fprintf (ptr_out, "%s %11d %12.7g %12.7g %12.7g %12.7g\n",
               sample, n_accept, cost, cost_temp, bias_accept, aver_weight);
    } else {
      fprintf (ptr_out, "%s %10d %12.7g %12.7g %12.7g %12.7g\n",
               sample, n_accept, cost, cost_temp, bias_accept, aver_weight);
    }
    prod = bias_accept;
    binsize = 1.0;
    for (ind = 0; ind < dim; ++ind) {
      fscanf_ret = fscanf (ptr_asa, "%s%d%lf%lf%lf%lf",
                           sample, &index, &param, &temp, &bias_gener,
                           &range);
      fprintf (ptr_out, "%s %11d %12.7g %12.7g %12.7g %12.7g\n", sample,
               index, param, temp, bias_gener, range);
      prod *= bias_gener;
      binsize *= range;
    }
    /* In this example, retrieve integrand from sampling function */
    sum += ((F_EXP (-cost) * binsize) / prod);
    norm += (binsize / prod);
  }
  sum /= norm;

  answer = 1.0;
  for (ind = 0; ind < dim; ++ind) {
    answer *= (0.5 * sqrt (3.14159265) * erf (2.0));
  }

  fprintf (ptr_out, "\n");
  fprintf (ptr_out, "sum = %12.7g, answer = %12.7g\n", sum, answer);
  fprintf (ptr_out, "n_samples = %d, norm = %12.7g\n", n_samples, norm);
  fflush (ptr_out);

}
#endif /* ASA_TEMPLATE_SAMPLE */
#if ASA_TEMPLATE_LIB
int
main ()
{
  double main_cost_value;
  double *main_cost_parameters;
  int main_exit_code;
  LONG_INT number_params;
  ALLOC_INT n_param;
  FILE *ptr_main;

#if INCL_STDOUT
  ptr_main = stdout;
#endif /* INCL_STDOUT */

  /* Note this assumes the *parameter_dimension = 4 */
  number_params = 4;

  if ((main_cost_parameters =
       (double *) calloc (number_params, sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "ASA_TEMPLATE_LIB main(): main_cost_parameters");
    Exit_USER (user_exit_msg);
    return (-2);
  }

  asa_seed (696969);            /* This is the default random seed. */
  asa_main (&main_cost_value, main_cost_parameters, &main_exit_code);

  fprintf (ptr_main, "main_exit_code = %d\n", main_exit_code);
  fprintf (ptr_main, "main_cost_value = %12.7g\n", main_cost_value);
  fprintf (ptr_main, "parameter\tvalue\n");
  for (n_param = 0; n_param < number_params; ++n_param) {
    fprintf (ptr_main,
#if INT_ALLOC
             "%d\t\t%12.7g\n",
#else
#if INT_LONG
             "%ld\t\t%12.7g\n",
#else
             "%d\t\t%12.7g\n",
#endif
#endif
             n_param, main_cost_parameters[n_param]);
  }

  free (main_cost_parameters);

  return (0);
/* NOTREACHED */
}
#endif /* ASA_TEMPLATE_LIB */

#if ADAPTIVE_OPTIONS
/* examples of possible entries in asa_adaptive_options file
User_Quench_Param_Scale,1,0.7
User_Quench_Cost_Scale,0,0.85
Cost_Parameter_Scale_Ratio,1.2
*/
#if HAVE_ANSI
void
adaptive_options (USER_DEFINES * USER_OPTIONS)
#else
void
adaptive_options (USER_OPTIONS)
     USER_DEFINES *USER_OPTIONS;
#endif /* HAVE_ANSI */
{
  int ndum, ndim;
  long int ldum;
  double ddum;
  char cdum[80];
  char line[200];
  char delim[5];
  FILE *ptr_adaptive;

  /* initialize */
  ndim = ndum = 0;
  ldum = 0;
  ddum = 0;
  delim[0] = ',';
  delim[1] = '\n';
  delim[2] = '\0';

  if ((ptr_adaptive = fopen ("asa_adaptive_options", "r")) == NULL) {
    return;
  }

  while (fgets (line, 200, ptr_adaptive) != NULL) {
    strcpy (cdum, strtok (line, delim));

    if (!strcmp (cdum, "Limit_Acceptances")) {
      ldum = (long int) atoi (strtok (NULL, delim));
      USER_OPTIONS->Limit_Acceptances = ldum;
    } else if (!strcmp (cdum, "Limit_Generated")) {
      ldum = (long int) atoi (strtok (NULL, delim));
      USER_OPTIONS->Limit_Generated = ldum;
    } else if (!strcmp (cdum, "Limit_Invalid_Generated_States")) {
      ndum = (int) atoi (strtok (NULL, delim));
      USER_OPTIONS->Limit_Invalid_Generated_States = ndum;
    } else if (!strcmp (cdum, "Accepted_To_Generated_Ratio")) {
      ddum = (double) atof (strtok (NULL, delim));
      USER_OPTIONS->Accepted_To_Generated_Ratio = ddum;
    } else if (!strcmp (cdum, "Cost_Precision")) {
      ddum = (double) atof (strtok (NULL, delim));
      USER_OPTIONS->Cost_Precision = ddum;
    } else if (!strcmp (cdum, "Maximum_Cost_Repeat")) {
      ndum = (int) atoi (strtok (NULL, delim));
      USER_OPTIONS->Maximum_Cost_Repeat = ndum;
    } else if (!strcmp (cdum, "Temperature_Ratio_Scale")) {
      ddum = (double) atof (strtok (NULL, delim));
      USER_OPTIONS->Temperature_Ratio_Scale = ddum;
    } else if (!strcmp (cdum, "Cost_Parameter_Scale_Ratio")) {
      ddum = (double) atof (strtok (NULL, delim));
      USER_OPTIONS->Cost_Parameter_Scale_Ratio = ddum;
    } else if (!strcmp (cdum, "Temperature_Anneal_Scale")) {
      ddum = (double) atof (strtok (NULL, delim));
      USER_OPTIONS->Temperature_Anneal_Scale = ddum;
    }
#if RATIO_TEMPERATURE_SCALES
    else if (!strcmp (cdum, "User_Temperature_Ratio")) {
      ndim = (int) atoi (strtok (NULL, delim));
      ddum = (double) atof (strtok (NULL, delim));
      USER_OPTIONS->User_Temperature_Ratio[ndim] = ddum;
    }
#endif
    else if (!strcmp (cdum, "Acceptance_Frequency_Modulus")) {
      ndum = (int) atoi (strtok (NULL, delim));
      USER_OPTIONS->Acceptance_Frequency_Modulus = ndum;
    } else if (!strcmp (cdum, "Generated_Frequency_Modulus")) {
      ndum = (int) atoi (strtok (NULL, delim));
      USER_OPTIONS->Generated_Frequency_Modulus = ndum;
    }
#if QUENCH_PARAMETERS
    else if (!strcmp (cdum, "User_Quench_Param_Scale")) {
      ndim = (int) atoi (strtok (NULL, delim));
      ddum = (double) atof (strtok (NULL, delim));
      USER_OPTIONS->User_Quench_Param_Scale[ndim] = ddum;
    }
#endif
#if QUENCH_COST
    else if (!strcmp (cdum, "User_Quench_Cost_Scale")) {
      ndim = (int) atoi (strtok (NULL, delim));
      ddum = (double) atof (strtok (NULL, delim));
      USER_OPTIONS->User_Quench_Cost_Scale[ndim] = ddum;
    }
#endif
    else if (!strcmp (cdum, "N_Accepted")) {
      ldum = (long int) atoi (strtok (NULL, delim));
      USER_OPTIONS->N_Accepted = ldum;
    } else if (!strcmp (cdum, "N_Generated")) {
      ldum = (long int) atoi (strtok (NULL, delim));
      USER_OPTIONS->N_Generated = ldum;
    }
  }
  fclose (ptr_adaptive);

  return;
}
#endif /* ADAPTIVE_OPTIONS */

#if ASA_FUZZY

/* This code is taken from
 * https://sites.google.com/site/stochasticglobaloptimization/home/fuzzy-asa
 * courtesy of the developer Hime Junior <hime@engineer.com> */

/* N.b.: Important parameters have been made adaptive within USER_OPTIONS,
 * but there are also eight define parameters below also can be changed */

/*
    As USER_OPTIONS->NoOfSamples regulates the size of arrays storing data about some best values found recently, 
    the user can freely change it. Please, notice that values higher than 100 or lower than 5, for instance, 
    could cause problems to the fuzzy controller's "reasoning".
*/

/* Initialization of fuzzy controller's data structures - should be called just before activations of asa() */
int
InitFuzzyASA (USER_DEFINES * USER_OPTIONS, ALLOC_INT NoOfDimensions)
{

  int index;

#if ASA_TEMPLATE
  USER_OPTIONS->NoOfSamples = 30;
  USER_OPTIONS->ThresholdDeviation = 0.5E-4;
  USER_OPTIONS->Threshold1 = -0.0001;
  USER_OPTIONS->Performance_Target = 0.1;
  USER_OPTIONS->Factor_a = 2;

#endif
  /* defaults */
  USER_OPTIONS->NoOfSamples = 30;
  USER_OPTIONS->ThresholdDeviation = 0.5E-4;
  USER_OPTIONS->Threshold1 = -0.0001;
  USER_OPTIONS->Performance_Target = 0.1;
  USER_OPTIONS->Factor_a = 2;

  ValMinLoc = log (1 / (1 + exp (-USER_OPTIONS->Factor_a)));

  if ((FuzzyValues =
       (double *) calloc (USER_OPTIONS->NoOfSamples + 1,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "InitFuzzyASA: FuzzyValues");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  if ((FuzzyMinima =
       (double *) calloc (USER_OPTIONS->NoOfSamples + 1,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "InitFuzzyASA: FuzzyValues");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  if ((auxV =
       (double *) calloc (USER_OPTIONS->NoOfSamples,
                          sizeof (double))) == NULL) {
    strcpy (user_exit_msg, "InitFuzzyASA: auxV");
    Exit_USER (user_exit_msg);
    return (-2);
  }

  if ((FuzzyParameters =
       (double **) calloc (USER_OPTIONS->NoOfSamples,
                           sizeof (double *))) == NULL) {
    strcpy (user_exit_msg, "InitFuzzyASA: *FuzzyParameters");
    Exit_USER (user_exit_msg);
    return (-2);
  }
  for (index = 0; index < USER_OPTIONS->NoOfSamples; ++index)
    if ((FuzzyParameters[index] =
         (double *) calloc (NoOfDimensions, sizeof (double))) == NULL) {
      strcpy (user_exit_msg, "InitFuzzyASA: FuzzyParameters");
      Exit_USER (user_exit_msg);
      return (-2);
    }

  return 0;
}

/* Release of fuzzy controller's data structures obtained previously - should be called just after returning of asa() */
void
CloseFuzzyASA (USER_DEFINES * USER_OPTIONS)
{

  int index;

  for (index = 0; index < USER_OPTIONS->NoOfSamples; ++index) {
    free (FuzzyParameters[index]);
  }
  free (FuzzyParameters);

  free (auxV);
  free (FuzzyMinima);
  free (FuzzyValues);
}

void
MeanAndDeviation (USER_DEFINES * USER_OPTIONS, double *Vector,
                  int NumberOfElements, double *Mean, double *Deviation)
{

  int i;
  double aux;

  *Mean = *Deviation = 0;

  for (i = 0; i < NumberOfElements; i++) {
    aux = Vector[i];
    *Mean += aux;
    *Deviation += aux * aux;
  }

  *Mean /= NumberOfElements;
  *Deviation /= NumberOfElements;
  *Deviation = *Deviation - (*Mean) * (*Mean);
  *Deviation = sqrt (fabs (*Deviation));
}

double
SubEnergy (USER_DEFINES * USER_OPTIONS, double InputValue, double Minimum)
{

  double valuhat, argulog;

  valuhat = InputValue - Minimum;
  argulog = 1 / (1 + exp (-(valuhat + USER_OPTIONS->Factor_a)));

  if (argulog == 0) {
    argulog = 1e-100;
  }

  return log (argulog);
}

double
DeltaFactor (USER_DEFINES * USER_OPTIONS, double MeanSub)
{
#define DELTATPOS       .3
#define DELTATNULO      0
#define DELTATNEG      -.3
#define INTER1          .1

  double MembershipMeanZero = 0;
  double MembershipMeanMedium = 0;
  double ResuCrisp;

/*
 RULE #1 - IF MeanSub is ZERO THEN decreasing rate is POSITIVE
 ( We are distant from present basic local minimum )
*/

  if (MeanSub > USER_OPTIONS->Threshold1) {
    MembershipMeanZero =
      (MeanSub - USER_OPTIONS->Threshold1) / (-USER_OPTIONS->Threshold1);
  }

/*
 RULE #2 - IF MeanSub is ValMinLoc ( FUZZY NUMBER )
                       THEN decreasing rate is POSITIVE
( We are in a region near the present basic local minimum )
*/

  if (MeanSub >= ValMinLoc && MeanSub <= ValMinLoc + INTER1) {
    MembershipMeanMedium = (ValMinLoc + INTER1 - MeanSub) / INTER1;
  }

  ResuCrisp =
    MembershipMeanZero * DELTATPOS + MembershipMeanMedium * DELTATPOS;

  return ResuCrisp;
}

void
AlterQuench (USER_DEFINES * USER_OPTIONS,
             int NoParam, double Mean, double Deviation)
{
#define Mult1 0.5
#define Mult2 0.2
#define Mult3 0.1
#define Mult4 0.15

  int i, j;
  double Delta, Meanaux, Deviationaux;

  Delta = DeltaFactor (USER_OPTIONS, Mean);
  if (USER_OPTIONS->User_Quench_Cost_Scale[0] < 100) {
    USER_OPTIONS->User_Quench_Cost_Scale[0] *= (1 + Mult1 * Delta);
  }

  for (i = 0; i < NoParam; i++) {
    if (USER_OPTIONS->User_Quench_Param_Scale[i] < 100) {
      USER_OPTIONS->User_Quench_Param_Scale[i] *= (1 + Mult2 * Delta);
    }
  }

  if (Deviation < USER_OPTIONS->ThresholdDeviation) {
    for (i = 0; i < NoParam; i++) {
      if (USER_OPTIONS->User_Quench_Param_Scale[i] < 100
          && USER_OPTIONS->User_Quench_Param_Scale[i] > Mult3) {
        for (j = 0; j < USER_OPTIONS->NoOfSamples; j++) {
          auxV[j] = FuzzyParameters[j][i];
        }

        MeanAndDeviation (USER_OPTIONS, auxV, USER_OPTIONS->NoOfSamples,
                          &Meanaux, &Deviationaux);
        USER_OPTIONS->User_Quench_Param_Scale[i] /= (1 + Mult4 * Delta *
                                                     exp (-Deviationaux));
      }
    }
  }

}

void
FuzzyControl (USER_DEFINES * USER_OPTIONS, double *x, double fvalue,
              ALLOC_INT dimensions)
{
  static double ActualPerformance, Mean, Deviation;
  static int IndVal = 0;
  int i, NoParam;

  IndVal++;
  NoParam = (int) dimensions;

  if (IndVal % (USER_OPTIONS->NoOfSamples + 1)) {
    FuzzyValues[IndVal] = fvalue;
    FuzzyMinima[IndVal] = *USER_OPTIONS->Best_Cost;

    for (i = 0; i < NoParam; i++) {
      FuzzyParameters[IndVal - 1][i] = USER_OPTIONS->Best_Parameters[i];        // Stores better results until now
    }

    return;
  }

  IndVal = 0;

  if (FuzzyMinima[1] != 0) {
    ActualPerformance =
      (FuzzyMinima[1] -
       FuzzyMinima[USER_OPTIONS->NoOfSamples]) / fabs (FuzzyMinima[1]);
  } else {
    return;
  }

  if (ActualPerformance > USER_OPTIONS->Performance_Target) {
    return;
  }

  for (i = 0; i < USER_OPTIONS->NoOfSamples; i++) {
    auxV[i] = SubEnergy (USER_OPTIONS, FuzzyValues[i + 1], FuzzyMinima[i + 1]); // Zero based
  }

  MeanAndDeviation (USER_OPTIONS, auxV, USER_OPTIONS->NoOfSamples, &Mean,
                    &Deviation);
  AlterQuench (USER_OPTIONS, NoParam, Mean, Deviation);

  return;
}
#endif /* ASA_FUZZY */

void
Exit_USER (char *statement)
{
#if INCL_STDOUT
  printf ("\n\n*** EXIT calloc failed *** %s\n\n", statement);
#else
  ;
#endif /* INCL_STDOUT */
}
