// Quadrotor hovering example

// This script is just to show how to use the library, the data for this example is not tuned for our Crazyflie demo. Check the firmware code for more details.

// - NSTATES = 12
// - NINPUTS = 4
// - NHORIZON = anything you want
// - tinytype = float if you want to run on microcontrollers
// States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
// phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
// check this paper for more details: https://ieeexplore.ieee.org/document/9326337
// Inputs: u1, u2, u3, u4 (motor thrust 0-1, order from Crazyflie)

#define NSTATES 12
#define NINPUTS 4

#define NHORIZON 10

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/tiny_api.hpp>
#include "problem_data/quadrotor_20hz_params.hpp"

#include <ctime>

extern "C"
{

    typedef Matrix<tinytype, NINPUTS, NHORIZON - 1> tiny_MatrixNuNhm1;
    typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
    typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

    int main()
    {
        std::clock_t start_quadrotor_hovering_tinyMPC, end_quadrotor_hovering_tinyMPC;
        start_quadrotor_hovering_tinyMPC = std::clock();

        TinySolver *solver;

        tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
        tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
        tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
        tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

        tinyMatrix x_min = tiny_MatrixNxNh::Constant(-5);
        tinyMatrix x_max = tiny_MatrixNxNh::Constant(5);
        tinyMatrix u_min = tiny_MatrixNuNhm1::Constant(-0.5);
        tinyMatrix u_max = tiny_MatrixNuNhm1::Constant(0.5);

        int status = tiny_setup(&solver,
                                Adyn, Bdyn, Q.asDiagonal(), R.asDiagonal(),
                                rho_value, NSTATES, NINPUTS, NHORIZON,
                                x_min, x_max, u_min, u_max, 1);

        // Update whichever settings we'd like
        solver->settings->max_iter = 100;

        // Alias solver->work for brevity
        TinyWorkspace *work = solver->work;

        // Initial state
        tiny_VectorNx x0;
        x0 << 0, 1, 0, 0.2, 0, 0, 0.1, 0, 0, 0, 0, 0;

        // Reference trajectory
        tiny_VectorNx Xref_origin;
        // Xref_origin << 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        Xref_origin << 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        work->Xref = Xref_origin.replicate<1, NHORIZON>();

        for (int k = 0; k < 70; ++k)
        {
            // printf("Timestep %2d\n:", k);

            // printf("tracking error at step %2d: %.4f\n", k, (x0 - work->Xref.col(1)).norm());
            // std::cout << "x : " << x0.transpose() << std::endl;

            // 1. Update measurement
            tiny_set_x0(solver, x0);

            // 2. Solve MPC problem
            tiny_solve(solver);
            // std::cout << "u : " << work->u.col(0).transpose() << std::endl;

            // 3. Simulate forward
            x0 = work->Adyn * x0 + work->Bdyn * work->u.col(0);
        }

        end_quadrotor_hovering_tinyMPC = std::clock();
        double elapsed_time_quadrotor_hovering_tinyMPC = double(end_quadrotor_hovering_tinyMPC - start_quadrotor_hovering_tinyMPC) / CLOCKS_PER_SEC;
        std::cout << std::endl;
        std::cout << "Elapsed time of solving the solution of quadrotor hovering via tinyMPC: \n"
                  << elapsed_time_quadrotor_hovering_tinyMPC << " seconds" << std::endl;
        std::cout << std::endl;

        return 0;
    }

} /* extern "C" */