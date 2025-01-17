// Quadrotor tracking example

// This script is just to show how to use the library, the data for this example is not tuned for our Crazyflie demo. Check the firmware code for more details.

// - NSTATES = 12
// - NINPUTS = 4
// - NHORIZON = anything you want
// - NTOTAL = 301 if using reference trajectory from trajectory_data/
// - tinytype = float if you want to run on microcontrollers
// States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
// phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
// check this paper for more details: https://ieeexplore.ieee.org/document/9326337
// Inputs: u1, u2, u3, u4 (motor thrust 0-1, order from Crazyflie)


#define NSTATES 12
#define NINPUTS 4

#define NHORIZON 10
#define NTOTAL 301

#include <iostream>
#include <tinympc/tiny_api.hpp>

#include <ctime>

extern "C" {

#include "problem_data/quadrotor_20hz_params.hpp"
#include "trajectory_data/quadrotor_20hz_y_axis_line.hpp"

typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

int main()
{
    std::clock_t start_quadrotor_tracking_tinyMPC, end_quadrotor_tracking_tinyMPC;
    start_quadrotor_tracking_tinyMPC = std::clock();

    std::cout << "Starting quadrotor tracking via tinyMPC..." << std::endl;

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

    // Map data from trajectory_data
    Matrix<tinytype, NSTATES, NTOTAL> Xref_total = Eigen::Map<Matrix<tinytype, NSTATES, NTOTAL>>(Xref_data);
    work->Xref = Xref_total.block<NSTATES, NHORIZON>(0, 0);

    // Initial state
    tiny_VectorNx x0;
    x0 = work->Xref.col(0);

    for (int k = 0; k < NTOTAL - NHORIZON; ++k)
    {
        std::cout << "tracking error: " << (x0 - work->Xref.col(1)).norm() << std::endl;

        // 1. Update measurement
        tiny_set_x0(solver, x0);

        // 2. Update reference
        work->Xref = Xref_total.block<NSTATES, NHORIZON>(0, k);

        // 3. Reset dual variables if needed
        work->y = tiny_MatrixNuNhm1::Zero();
        work->g = tiny_MatrixNxNh::Zero();

        // 4. Solve MPC problem
        tiny_solve(solver);

        // 5. Simulate forward
        // 使用临时变量减少重复计算
        const tinyMatrix &u0 = work->u.col(0);
        x0 = work->Adyn * x0 + work->Bdyn * u0;
    }

    end_quadrotor_tracking_tinyMPC = std::clock();
    double elapsed_time_quadrotor_tracking_tinyMPC = double(end_quadrotor_tracking_tinyMPC - start_quadrotor_tracking_tinyMPC) / CLOCKS_PER_SEC;
    std::cout << std::endl;
    std::cout << "Elapsed time of solving the solution of quadrotor tracking via tinyMPC: \n"
              << elapsed_time_quadrotor_tracking_tinyMPC << " seconds" << std::endl;
    std::cout << std::endl;

    return 0;
}

} /* extern "C" */