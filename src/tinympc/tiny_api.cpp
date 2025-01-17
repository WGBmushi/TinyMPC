#include "tiny_api.hpp"
#include "tiny_api_constants.hpp"

#include <iostream>
#include <ctime>

#ifdef __cplusplus
extern "C"
{
#endif

    using namespace Eigen;
    IOFormat TinyApiFmt(4, 0, ", ", "\n", "[", "]");

    static int check_dimension(std::string matrix_name, std::string rows_or_columns, int actual, int expected)
    {
        if (actual != expected)
        {
            std::cout << matrix_name << " has " << actual << " " << rows_or_columns << ". Expected " << expected << "." << std::endl;
            return 1;
        }
        return 0;
    }

    int tiny_setup(TinySolver **solverp,
                   tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R,
                   tinytype rho, int nx, int nu, int N,
                   tinyMatrix x_min, tinyMatrix x_max, tinyMatrix u_min, tinyMatrix u_max,
                   int verbose)
    {

        TinySolution *solution = new TinySolution();
        TinyCache *cache = new TinyCache();
        TinySettings *settings = new TinySettings();
        TinyWorkspace *work = new TinyWorkspace();
        TinySolver *solver = new TinySolver();

        solver->solution = solution;
        solver->cache = cache;
        solver->settings = settings;
        solver->work = work;

        *solverp = solver;

        // Initialize solution
        solution->iter = 0;
        solution->solved = 0;
        solution->x = tinyMatrix::Zero(nx, N);
        solution->u = tinyMatrix::Zero(nu, N - 1);

        // Initialize settings
        tiny_set_default_settings(settings);

        // Initialize workspace
        work->nx = nx;
        work->nu = nu;
        work->N = N;

        // Make sure arguments are the correct shapes
        int status = 0;
        status |= check_dimension("State transition matrix (A)", "rows", Adyn.rows(), nx);
        status |= check_dimension("State transition matrix (A)", "columns", Adyn.cols(), nx);
        status |= check_dimension("Input matrix (B)", "rows", Bdyn.rows(), nx);
        status |= check_dimension("Input matrix (B)", "columns", Bdyn.cols(), nu);
        status |= check_dimension("State stage cost (Q)", "rows", Q.rows(), nx);
        status |= check_dimension("State stage cost (Q)", "columns", Q.cols(), nx);
        status |= check_dimension("State input cost (R)", "rows", R.rows(), nu);
        status |= check_dimension("State input cost (R)", "columns", R.cols(), nu);
        status |= check_dimension("Lower state bounds (x_min)", "rows", x_min.rows(), nx);
        status |= check_dimension("Lower state bounds (x_min)", "cols", x_min.cols(), N);
        status |= check_dimension("Lower state bounds (x_max)", "rows", x_max.rows(), nx);
        status |= check_dimension("Lower state bounds (x_max)", "cols", x_max.cols(), N);
        status |= check_dimension("Lower input bounds (u_min)", "rows", u_min.rows(), nu);
        status |= check_dimension("Lower input bounds (u_min)", "cols", u_min.cols(), N - 1);
        status |= check_dimension("Lower input bounds (u_max)", "rows", u_max.rows(), nu);
        status |= check_dimension("Lower input bounds (u_max)", "cols", u_max.cols(), N - 1);

        work->x = tinyMatrix::Zero(nx, N);
        work->u = tinyMatrix::Zero(nu, N - 1);

        work->q = tinyMatrix::Zero(nx, N);
        work->r = tinyMatrix::Zero(nu, N - 1);

        work->p = tinyMatrix::Zero(nx, N);
        work->d = tinyMatrix::Zero(nu, N - 1);

        work->v = tinyMatrix::Zero(nx, N);
        work->vnew = tinyMatrix::Zero(nx, N);
        work->z = tinyMatrix::Zero(nu, N - 1);
        work->znew = tinyMatrix::Zero(nu, N - 1);

        work->g = tinyMatrix::Zero(nx, N);
        work->y = tinyMatrix::Zero(nu, N - 1);

        work->Q = (Q + rho * tinyMatrix::Identity(nx, nx)).diagonal();
        work->R = (R + rho * tinyMatrix::Identity(nu, nu)).diagonal();
        work->Adyn = Adyn;
        work->Bdyn = Bdyn;

        work->x_min = x_min;
        work->x_max = x_max;
        work->u_min = u_min;
        work->u_max = u_max;

        work->Xref = tinyMatrix::Zero(nx, N);
        work->Uref = tinyMatrix::Zero(nu, N - 1);

        work->Qu = tinyVector::Zero(nu);

        work->primal_residual_state = 0;
        work->primal_residual_input = 0;
        work->dual_residual_state = 0;
        work->dual_residual_input = 0;
        work->status = 0;
        work->iter = 0;

        // Initialize cache
        status = tiny_precompute_and_set_cache(cache, Adyn, Bdyn, work->Q.asDiagonal(), work->R.asDiagonal(), nx, nu, rho, verbose);
        if (status)
        {
            return status;
        }

        return 0;
    }

    int tiny_precompute_and_set_cache(TinyCache *cache,
                                      tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R,
                                      int nx, int nu, tinytype rho, int verbose)
    {

        if (!cache)
        {
            std::cout << "Error in tiny_precompute_and_set_cache: cache is nullptr" << std::endl;
            return 1;
        }

        // Update by adding rho * identity matrix to Q, R
        tinyMatrix Q1 = Q + rho * tinyMatrix::Identity(nx, nx);
        tinyMatrix R1 = R + rho * tinyMatrix::Identity(nu, nu);

        // Printing
        if (verbose)
        {
            std::cout << "A = " << Adyn.format(TinyApiFmt) << std::endl;
            std::cout << "B = " << Bdyn.format(TinyApiFmt) << std::endl;
            std::cout << "Q = " << Q1.format(TinyApiFmt) << std::endl;
            std::cout << "R = " << R1.format(TinyApiFmt) << std::endl;
            std::cout << "rho = " << rho << std::endl;
        }

        tinyMatrix Ktp1 = tinyMatrix::Zero(nu, nx);
        tinyMatrix Ptp1 = rho * tinyMatrix::Ones(nx, 1).array().matrix().asDiagonal();

        std::clock_t start_Riccati_recursion, end_Riccati_recursion;
        if (verbose)
        {
            start_Riccati_recursion = std::clock();
        }

        // Riccati recursion initiated from the terminal state
        tinyMatrix Kinf = tinyMatrix::Zero(nu, nx);
        tinyMatrix Pinf = tinyMatrix::Zero(nx, nx);

        // Riccati recursion to get Kinf, Pinf
        for (int i = 0; i < 1000; i++)
        {
            Kinf = (R1 + Bdyn.transpose() * Ptp1 * Bdyn).inverse() * Bdyn.transpose() * Ptp1 * Adyn;
            Pinf = Q1 + Adyn.transpose() * Ptp1 * (Adyn - Bdyn * Kinf);
            // if Kinf converges, break
            if ((Kinf - Ktp1).cwiseAbs().maxCoeff() < 1e-5)
            {
                if (verbose)
                {
                    std::cout << "Kinf converged after " << i + 1 << " iterations" << std::endl;
                }
                break;
            }
            Ktp1 = Kinf;
            Ptp1 = Pinf;
        }

        /*
        // Warm start Kinf and Pinf
        // Significant improvements. For quadrotor-related projects, the Riccati iteration achieves an acceleration of approximately 80%.

        // Riccati recursion to get Kinf, Pinf
        tinyMatrix Kinf(nu, nx);
        tinyMatrix Pinf(nx, nx);
        for (int i = 0; i < 1000; i++)
        {
            if (i == 0)
            {
                // Initial guess for Kinf and Pinf
                Kinf << -0.08904, 0.06347, 1.07, -0.3647, -0.5458, -2.089, -0.07684, 0.05341, 0.5225, -0.0348, -0.05539, -0.5286,
                    0.0849, 0.02487, 1.07, -0.09954, 0.5238, 2.089, 0.07346, 0.01851, 0.5225, -0.005326, 0.05347, 0.5283,
                    0.002888, -0.02902, 1.07, 0.1215, -0.06214, -2.087, -0.001955, -0.0219, 0.5225, 0.007242, -0.01353, -0.5275,
                    0.001251, -0.05932, 1.07, 0.3427, 0.08409, 2.088, 0.005337, -0.05002, 0.5225, 0.03289, 0.01544, 0.5278;

                Pinf << 1792, -0.2446, 0.0002435, 1.249, 1719, 16.94, 544.8, -0.2001, 0.0001072, 0.09271, 19.58, 3.218,
                    -0.2446, 1791, -1.069e-06, -1717, -1.249, -6.774, -0.2001, 544.4, -8.683e-07, -19.39, -0.09273, -1.287,
                    0.0002435, -1.069e-06, 1074, 6.008e-06, 0.001138, 0.0004983, 0.0001989, -9.098e-07, 95.38, 4.683e-07, 1.444e-05, 2.225e-05,
                    1.249, -1717, 6.008e-06, 7723, 7.408, 43.66, 1.08, -1313, 5.469e-06, 91.52, 0.6307, 9.029,
                    1719, -1.249, 0.001138, 7.408, 7738, 109.1, 1315, -1.081, 0.0006945, 0.6307, 92.78, 22.57,
                    16.94, -6.774, 0.0004983, 43.66, 109.1, 4146, 15.07, -6.029, 0.0004277, 4.098, 10.24, 192.2,
                    544.8, -0.2001, 0.0001989, 1.08, 1315, 15.07, 349.5, -0.1673, 9.535e-05, 0.08368, 15.3, 2.938,
                    -0.2001, 544.4, -9.098e-07, -1313, -1.081, -6.029, -0.1673, 349.2, -7.684e-07, -15.13, -0.08369, -1.175,
                    0.0001072, -8.683e-07, 95.38, 5.469e-06, 0.0006945, 0.0004277, 9.535e-05, -7.684e-07, 51.19, 4.85e-07, 9.362e-06, 2.235e-05,
                    0.09271, -19.39, 4.683e-07, 91.52, 0.6307, 4.098, 0.08368, -15.13, 4.85e-07, 13.17, 0.06845, 1.013,
                    19.58, -0.09273, 1.444e-05, 0.6307, 92.78, 10.24, 15.3, -0.08369, 9.362e-06, 0.06845, 13.31, 2.531,
                    3.218, -1.287, 2.225e-05, 9.029, 22.57, 192.2, 2.938, -1.175, 2.235e-05, 1.013, 2.531, 53.2;
            }
            else
            {
                Kinf = (R1 + Bdyn.transpose() * Ptp1 * Bdyn).inverse() * Bdyn.transpose() * Ptp1 * Adyn;
                Pinf = Q1 + Adyn.transpose() * Ptp1 * (Adyn - Bdyn * Kinf);
            }
            // if Kinf converges, break
            if ((Kinf - Ktp1).cwiseAbs().maxCoeff() < 1e-5)
            {
                if (verbose)
                {
                    std::cout << "Kinf converged after " << i + 1 << " iterations" << std::endl;
                }
                break;
            }
            Ktp1 = Kinf;
            Ptp1 = Pinf;
        }
        */

        if (verbose)
        {
            end_Riccati_recursion = std::clock();
            double elapsed_time_Riccati_recursion = double(end_Riccati_recursion - start_Riccati_recursion) / CLOCKS_PER_SEC;
            std::cout << std::endl;
            std::cout << "Elapsed time of solving the Riccati recursion: \n"
                      << elapsed_time_Riccati_recursion << " seconds" << std::endl;
            std::cout << std::endl;
        }

        // Compute cached matrices
        tinyMatrix Quu_inv = (R1 + Bdyn.transpose() * Pinf * Bdyn).inverse();
        tinyMatrix AmBKt = (Adyn - Bdyn * Kinf).transpose();

        if (verbose)
        {
            std::cout << "Kinf = " << Kinf.format(TinyApiFmt) << std::endl;
            std::cout << "Pinf = " << Pinf.format(TinyApiFmt) << std::endl;
            std::cout << "Quu_inv = " << Quu_inv.format(TinyApiFmt) << std::endl;
            std::cout << "AmBKt = " << AmBKt.format(TinyApiFmt) << std::endl;

            std::cout << "\nPrecomputation finished!\n"
                      << std::endl;
        }

        cache->rho = rho;
        cache->Kinf = Kinf;
        cache->Pinf = Pinf;
        cache->Quu_inv = Quu_inv;
        cache->AmBKt = AmBKt;

        return 0; // return success
    }

    int tiny_solve(TinySolver *solver)
    {
        return solve(solver);
    }

    int tiny_update_settings(TinySettings *settings, tinytype abs_pri_tol, tinytype abs_dua_tol,
                             int max_iter, int check_termination,
                             int en_state_bound, int en_input_bound)
    {
        if (!settings)
        {
            std::cout << "Error in tiny_update_settings: settings is nullptr" << std::endl;
            return 1;
        }
        settings->abs_pri_tol = abs_pri_tol;
        settings->abs_dua_tol = abs_dua_tol;
        settings->max_iter = max_iter;
        settings->check_termination = check_termination;
        settings->en_state_bound = en_state_bound;
        settings->en_input_bound = en_input_bound;
        return 0;
    }

    int tiny_set_default_settings(TinySettings *settings)
    {
        if (!settings)
        {
            std::cout << "Error in tiny_set_default_settings: settings is nullptr" << std::endl;
            return 1;
        }
        settings->abs_pri_tol = TINY_DEFAULT_ABS_PRI_TOL;
        settings->abs_dua_tol = TINY_DEFAULT_ABS_DUA_TOL;
        settings->max_iter = TINY_DEFAULT_MAX_ITER;
        settings->check_termination = TINY_DEFAULT_CHECK_TERMINATION;
        settings->en_state_bound = TINY_DEFAULT_EN_STATE_BOUND;
        settings->en_input_bound = TINY_DEFAULT_EN_INPUT_BOUND;
        return 0;
    }

    int tiny_set_x0(TinySolver *solver, tinyVector x0)
    {
        if (!solver)
        {
            std::cout << "Error in tiny_set_x0: solver is nullptr" << std::endl;
            return 1;
        }
        if (x0.rows() != solver->work->nx)
        {
            perror("Error in tiny_set_x0: x0 is not the correct length");
        }
        solver->work->x.col(0) = x0;
        return 0;
    }

    int tiny_set_x_ref(TinySolver *solver, tinyMatrix x_ref)
    {
        if (!solver)
        {
            std::cout << "Error in tiny_set_x_ref: solver is nullptr" << std::endl;
            return 1;
        }
        int status = 0;
        status |= check_dimension("State reference trajectory (x_ref)", "rows", x_ref.rows(), solver->work->nx);
        status |= check_dimension("State reference trajectory (x_ref)", "columns", x_ref.cols(), solver->work->N);
        solver->work->Xref = x_ref;
        return 0;
    }

    int tiny_set_u_ref(TinySolver *solver, tinyMatrix u_ref)
    {
        if (!solver)
        {
            std::cout << "Error in tiny_set_u_ref: solver is nullptr" << std::endl;
            return 1;
        }
        int status = 0;
        status |= check_dimension("Control/input reference trajectory (u_ref)", "rows", u_ref.rows(), solver->work->nu);
        status |= check_dimension("Control/input reference trajectory (u_ref)", "columns", u_ref.cols(), solver->work->N - 1);
        solver->work->Uref = u_ref;
        return 0;
    }

#ifdef __cplusplus
}
#endif