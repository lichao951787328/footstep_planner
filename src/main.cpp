#include <iostream>
#include <vector>
#include <numeric> // For std::iota
#include "osqp.h"  // OSQP library header
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <matplotlibcpp.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace std;
#define degree2rad(x) x * M_PI/180.0;
struct Footstep
{
    bool is_left;
    double x, y, yaw;
    // 此处的变量个数应该与k相同
    vector<Eigen::Vector2d> u{10};
};


Eigen::Matrix4d get_autonomous_alip_matrix_A(double H_com, double mass, double g) 
{
    Eigen::Matrix4d A_c_autonomous;
    A_c_autonomous << 0, 0, 0, 1 / (mass * H_com),
                      0, 0, -1 / (mass * H_com), 0,
                      0, -mass * g, 0, 0,
                      mass * g, 0, 0, 0;
    return A_c_autonomous;
}


// 单脚支撑期内的两个矩阵
std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 2>> get_alip_matrices_with_input(double H_com, double mass, double g, double T_ss_dt) 
{
    Eigen::Matrix4d A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g);

    Eigen::Matrix<double, 4, 2> B_c_input_effect;
    B_c_input_effect.leftCols(1) = Eigen::Vector4d(0, 0, 1, 0); // roll L
    B_c_input_effect.rightCols(1) = Eigen::Vector4d(0, 0, 0, 1); // roll R

    Eigen::Matrix4d A_d_for_mpc = (A_c_autonomous * T_ss_dt).exp(); // 摆动期内，单个dot周期内的矩阵
    Eigen::Matrix<double, 4, 2> B_d_for_mpc;
    // This is valid if A_c is invertible
    if (std::abs(A_c_autonomous.determinant()) < 1e-9) 
    {
         std::cerr << "Warning: A_c_autonomous is singular or near-singular in get_alip_matrices_with_input. Using numerical integration for B_d." << std::endl;
        // Fallback or more robust method for singular A_c
        // For simple Euler integration of B: B_d approx A_c_autonomous.inverse() * (A_d_for_mpc - I) * B_c is not robust
        // A simple approximation: B_d = B_c_input_effect * T_ss_dt (first order hold for B_c)
        // Or integrate exp(A*s)*B from 0 to T_ss_dt.
        // The Python code's inv approach assumes invertibility. Let's try it but warn.
        // A more robust way to compute int_0^T exp(As) ds B can be done using the matrix exponential of an augmented matrix.
        // M = [[A, B], [0, 0]] -> exp(M*T) = [[Ad, Bd_int], [0, I]] where Bd_int = int_0^T exp(As)B ds
        // However, the python code uses A_c_inv * (A_d - I) * B_c
        B_d_for_mpc = A_c_autonomous.colPivHouseholderQr().solve((A_d_for_mpc - Eigen::Matrix4d::Identity()) * B_c_input_effect);

    } 
    else 
    {
      B_d_for_mpc = A_c_autonomous.inverse() * (A_d_for_mpc - Eigen::Matrix4d::Identity()) * B_c_input_effect;
    }
    return {A_d_for_mpc, B_d_for_mpc};
}

// 双脚支撑期内的两个矩阵
std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 3>> get_alip_reset_map_matrices_detailed(double T_ds, double H_com, double mass, double g) 
{
    // 3x⁵ - 2x⁴ + 5x³ - x² + 7x - 4
    Eigen::Matrix4d A_c = get_autonomous_alip_matrix_A(H_com, mass, g);
    Eigen::Matrix4d Ar_ds = (A_c * T_ds).exp();

    Eigen::Matrix<double, 4, 2> B_CoP_for_Bds;
    B_CoP_for_Bds << 0, 0,
                     0, 0,
                     0, mass * g,
                     -mass * g, 0;

    Eigen::Matrix<double, 4, 2> B_ds = Eigen::Matrix<double, 4, 2>::Zero();
    if (std::abs(A_c.determinant()) < 1e-9 || std::abs(Ar_ds.determinant()) < 1e-9) 
    {
        std::cerr << "Warning: A_c or Ar_ds is singular in get_alip_reset_map_matrices_detailed. B_ds set to zero." << std::endl;
    } 
    else
    {
        Eigen::Matrix4d A_c_inv = A_c.inverse();
        Eigen::Matrix4d Ar_ds_inv = Ar_ds.inverse();
        B_ds = Ar_ds * A_c_inv * ((1.0/T_ds) * A_c_inv * (Eigen::Matrix4d::Identity() - Ar_ds_inv) - Ar_ds_inv) * B_CoP_for_Bds;
    }

    Eigen::Matrix<double, 4, 3> B_fp;
    B_fp << 1, 0, 0,
            0, 1, 0,
            0, 0, 0,
            0, 0, 0;

    Eigen::Matrix<double, 4, 3> B_ds_padded = Eigen::Matrix<double, 4, 3>::Zero();
    B_ds_padded.block<4,2>(0,0) = B_ds;

    Eigen::Matrix<double, 4, 3> B_r = B_ds_padded + B_fp;
    return {Ar_ds, B_r};
}

double polynomial(Eigen::Vector<double, 6> & polynomial_param, double x) 
{
    return polynomial_param(0) * std::pow(x, 5) + polynomial_param(1) * std::pow(x, 4) + polynomial_param(2) * std::pow(x, 3) + polynomial_param(3) * std::pow(x, 2) + polynomial_param(4) * x + polynomial_param(5);
}

// 根据机器人的运动能力决定, 返回时默认左侧为第一个，右侧为第二个，后续使用时注意因左右带来的区别
std::pair<Eigen::Vector2d, Eigen::Vector2d> computerCircle(double x, double y, double yaw, bool is_left)
{
    Eigen::Vector2d short_v(0, 0.1, 0);
    Eigen::Vector2d long_v(0, 0.3, 0);
    
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 0) = cos(yaw); T(0, 0) = - sin(yaw);
    T(1, 0) = sin(yaw); T(1, 1) = cos(yaw);
    T(0, 3) = x;
    T(1, 3) = y;
    if (is_left)
    {
        // short_v
        Eigen::Vector3d v1 = T.block<3,3>(0,0) * short_v + T.block<3,1>(0,3);

        // -long_v
        Eigen::Vector3d v2 = T.block<3,3>(0,0) * (- long_v) + T.block<3,1>(0,3);
        return std::make_pair(Eigen::Vector2d(v1.x() + x, v1.y() + y), Eigen::Vector2d(v2.x() + x, v2.y() + y));
    }
    else
    {
        // -short_v
        // long_v
        Eigen::Vector3d v1 = T.block<3,3>(0,0) * (- short_v) + T.block<3,1>(0,3);
        Eigen::Vector3d v2 = T.block<3,3>(0,0) * ( long_v) + T.block<3,1>(0,3);
        return std::make_pair(Eigen::Vector2d(v2.x() + x, v2.y() + y), Eigen::Vector2d(v1.x() + x, v1.y() + y));
    }
}




int main(int argc, char** argv)
{
    // 求解变量：落脚点N个，每个为x、y、theta。每个摆动周期内有k个分段，共包含k-1个u，一共有N个摆动周期。

    // 控制端返回的机器人状态，主要包括机器人摆动脚的状态、当前机器人相对于摆动起始时刻的进程。如果在双脚支撑期，应该反馈双脚位置，以及两只脚哪只是支撑脚。
    // 需要与控制端进行沟通

    // 测试时，假设是以双脚支撑期为初始状态来进行规划，类似于全局落脚点规划
    // 机器人本体特性，H_com、mass、g
    double H_com = 0.9, mass = 60, g =9.81;
    double swing_t = 0.75, double_support = 0.15;
    Eigen::Vector4d x_initial; // 初始状态，可以为机器人赋一个固定的初始状态 ****来源于控制端反馈
    Footstep footstep_initial; // 跟u没有关系 cdcdcdcdv ccscdcdc    vdfvf
    footstep_initial.is_left = true;
    footstep_initial.x = 0;
    footstep_initial.y = 0.1;
    footstep_initial.yaw = 0.0; 

    double T_s2s_cycle = swing_t + double_support;
    // 求解前，N和k是提前设定好的
    
    int k = 10;
    
    double T_ss_dt = swing_t/(k - 1);

    

    Eigen::Vector<double, 6> polynomial_param;
    polynomial_param<<3, -2, 5, -1, 7, -4;

    /**
     * 画曲线图使用的参数
     */
    double start_x = -0.2;
    double end_x = 4.0;              
    double step = 0.01;

    vector<double> x_data;
    vector<double> y_data;

    for (double x = start_x; x <= end_x; x += step) {
        x_data.push_back(x);
        y_data.push_back(polynomial(polynomial_param, x));
    }

    // 根据多项式曲线，生成局部路径点
    // 终点位置 x,y,yaw
    double x_goal = 4.0;
    double y_goal = polynomial(polynomial_param, x_goal);

    // 以前面的某一点来计算切线方向
    double delta_x = 0.01;
    double delta_y = polynomial(polynomial_param, x_goal + delta_x) - polynomial(polynomial_param, x_goal);
    double yaw_goal = atan2(delta_y, delta_x);
    // 在终点位置，机器人的速度为0，因此，另外终点时刻的角动量为0
    Eigen::Vector4d x_goal(x_goal, y_goal, 0, 0);
    // to do list: 终点的yaw应该根据切线方向来确定
    // double yaw_goal = 0;

    /**
     * 此时存在一个假设：机器人双脚都是以双脚支撑期结束时作为开始。在这种情况下，规划的落脚点数与整个规划过程中机器人的状态数存在一个固定的数学关系
     */
    // 根据距离来设置一个合适的N,假设每步合适的距离为0.35m.
    int N = std::ceil((x_goal * x_goal + y_goal * y_goal) / (0.35 * 0.35)); // 保证N大于2，需要根据实际情况进行调整
    
    vector<Footstep> footsteps{N};

    // 整个状态 ---—|---—|---—
    // 以双脚支撑期的末尾为状态起点，那N步的最后一个状态还并没有表示
    vector<vector<Eigen::Vector4d>> x_state{N}; 
    for (auto & state_step : x_state)
    {
        state_step.resize(k + 1);
    }
    Eigen::Vector4d x_state_last;

    // *****************
    // 目标函数，代价最小包括
    // 代价1：踝关节力矩最小，保证行走稳定性
    double torque_cost = 0;
    Eigen::Matrix2d Q = Eigen::Matrix2d::Identity() * 100;
    for (auto & footstep : footsteps)
    {
        for (auto & u :footstep.u)
        {
            torque_cost += u.transpose() * Q * u;
        }
    }

    // 代价2：靠近终点 【x, y, 0, 0】
    double goal_cost = 0;
    double lamda_goal = 400;

    // 代价3：转角的均匀性
    double turn_cost = 0;
    double lamda_turn = 50;
    for (size_t i = 0; i < footsteps.size(); i++)
    {
        // 两步的转角不会很大，如果稳定的朝一个方向转弯，好像这个约束并没有实际意义，因为在某一步多转一点或转向均匀分布，是一样的。但是这一项可以避免反复转来转去的情况
        // turn_cost += abs(footsteps.at(i + 1).yaw - footsteps.at(i).yaw) * lamda_turn;

        // 所有的转角尽量均匀
        if (i == 0)
        {
            turn_cost += abs(footsteps.at(i).yaw - footstep_initial.yaw) * lamda_turn;
        }
        else
        {
            turn_cost += abs(footsteps.at(i).yaw - footsteps.at(i - 1).yaw) * lamda_turn;
        }
    }

    for (size_t i = 0; i < footsteps.size() - 1; i++)
    {
        if (i == 0) // 这个成为一个线性
        {
            turn_cost += abs((footsteps.at(i + 1).yaw - footsteps.at(i).yaw) - (footsteps.at(i).yaw - footstep_initial.yaw)) * lamda_turn;
        }
        else // 这个成为一个二次项
        {
            turn_cost += abs((footsteps.at(i + 1).yaw - footsteps.at(i).yaw) - (footsteps.at(i).yaw - footsteps.at(i - 1).yaw)) * lamda_turn;
        }
    }
    
    // 代价4：步态均匀性，不会出现前一步步长很大，后一步步长很小
    double step_cost = 0;
    double lamda_step = 80;
    // for (size_t i = 0; i < footsteps.size() - 2; i++)
    // {
    //     step_cost += Eigen::Vector2d(footsteps.at(i + 2).x - footsteps.at(i + 1).x, footsteps.at(i + 2).y - footsteps.at(i + 1).y).norm() * lamda_step;
    // }
    for (size_t i = 0; i < footsteps.size() - 1; i++)
    {
        if (i == 0)
        {
            double norm1 = Eigen::Vector2d(footsteps.at(i + 1).x - footsteps.at(i).x, footsteps.at(i + 1).y - footsteps.at(i).y).norm();
            double norm2 = Eigen::Vector2d(footsteps.at(i).x - footstep_initial.x, footsteps.at(i).y - footstep_initial.y).norm();
            step_cost += abs(norm1 -  norm2);
        }
        else
        {
            double norm1 = Eigen::Vector2d(footsteps.at(i + 1).x - footsteps.at(i).x, footsteps.at(i + 1).y - footsteps.at(i).y).norm();
            double norm2 = Eigen::Vector2d(footsteps.at(i).x - footsteps.at(i - 1).x, footsteps.at(i).y - footsteps.at(i - 1).y).norm();
            step_cost += abs(norm1 -  norm2);
        }
    }
    
    double total_cost = torque_cost + goal_cost + turn_cost + step_cost;

    // 约束的形式

    // 约束1：初始状态约束。等式约束
    x_state.front().front() = x_initial;

    // 约束1：ALIP模型约束。等式约束
    // 对着公式再检查一遍 to do list
    auto [A_d_mpc, B_d_mpc_vec] = get_alip_matrices_with_input(H_com, mass, g, T_ss_dt);
    Eigen::Matrix<double, 4, 2> B_d_mpc = B_d_mpc_vec; // Convert 4x1 vector to 4x1 matrix for consistency
    auto [Ar_reset, Br_reset_delta_p] = get_alip_reset_map_matrices_detailed(double_support, H_com, mass, g);
    Eigen::Matrix4d A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g);
    Eigen::Matrix4d A_d_autonomous_knot = (A_c_autonomous * T_ss_dt).exp();
    Eigen::Matrix4d A_s2s_autonomous_cycle = (A_c_autonomous * T_s2s_cycle).exp();
    // 前N-1步
    for (size_t i = 0; i < x_state.size() - 1; i++)
    {
        // 摆动周期部分
        for (size_t j = 0; j < k; j++) // 每个
        {
            x_state.at(i).at(j + 1) = A_d_mpc * x_state.at(i).at(j) + B_d_mpc * footsteps.at(i).u.at(j);
        }
        // 双脚支撑期部分
        if (i == 0)
        {
            x_state.at(i + 1).at(0) = Ar_reset * x_state.at(i).back() + Br_reset_delta_p * Eigen::Vector3d(footsteps.at(i).x - footstep_initial.x, footsteps.at(i).y - footstep_initial.y, 0); // 假设是平地
        }
        else
        {
            x_state.at(i + 1).at(0) = Ar_reset * x_state.at(i).back() + Br_reset_delta_p * Eigen::Vector3d(footsteps.at(i).x - footsteps.at(i - 1).x, footsteps.at(i).y - footsteps.at(i - 1).y, 0); 
        }
    }
    // 对最后一步进行特殊处理
    for (size_t i = 0; i < k; i++)
    {
        x_state.back().at(i + 1) = A_d_mpc * x_state.back().at(i) + B_d_mpc * footsteps.back().u.at(i);
    }
    x_state_last = Ar_reset * x_state.back().back() + Br_reset_delta_p * Eigen::Vector3d(footsteps.at(N -1).x - footsteps.at(N - 2).x, footsteps.at(N - 1).y - footsteps.at(N - 2).y, 0);
    
    Eigen::Vector2d u_max;
    // 约束3：力矩大小幅值约束。不等式约束
    for (auto & footstep : footsteps)
    {
        for (auto & u : footstep.u)
        {
            // 定义约束
            // abs(u) <= u_max;
            abs(u(0)) <= u_max(0);
            abs(u(1)) <= u_max(1);
        }   
    }

    double yaw_max = degree2rad(10);
    // 约束4：转角最大约束。不等式约束
    for (size_t i = 0; i < footsteps.size() - 1; i++)
    {
        abs(footsteps.at(i + 1).yaw - footsteps.at(i).yaw) <= yaw_max;
    }

    double r1, r2;
    // 约束5：运动学可达性约束。不等式约束。（三角函数可简化为一阶泰勒展开，前提是相邻角的转角较小，小于10度）
    for (size_t i = 0; i < footsteps.size(); i++)
    {
        if (i == 0)
        {
            auto circles = computerCircle(footstep_initial.x, footstep_initial.y, footstep_initial.yaw, footstep_initial.is_left);
                // 增加约束
            Eigen::Vector2d v1(footsteps.at(0).x - circles.first.x(), footsteps.at(0).y - circles.first.y());
            Eigen::Vector2d v2(footsteps.at(0).x - circles.second.x(), footsteps.at(0).y - circles.second.y());
            if (footstep_initial.is_left)
            {
                v1.norm() <= r1; // 外侧圆
                v2.norm() <= r2; // 外侧圆
            }
            else
            {
                v1.norm() <= r2; // 外侧圆
                v2.norm() <= r1; // 外侧圆
            }
        }
        else
        {
            auto circles = computerCircle(footsteps.at(i - 1).x, footsteps.at(i - 1).y, footsteps.at(i - 1).yaw, footsteps.at(i - 1).is_left);
            Eigen::Vector2d v1(footsteps.at(i).x - circles.first.x(), footsteps.at(i).y - circles.first.y());
            Eigen::Vector2d v2(footsteps.at(i).x - circles.second.x(), footsteps.at(i).y - circles.second.y());
            if (footsteps.at(i - 1).is_left)
            {
                v1.norm() <= r1; // 外侧圆
                v2.norm() <= r2; // 外侧圆
            }
            else
            {
                v1.norm() <= r2; // 外侧圆
                v2.norm() <= r1; // 外侧圆
            }
        }
    }

    // 约束6：相邻落脚点中点位于局部轨迹附近。不等式约束。这样就不用考虑避障问题，直接跟踪局部轨迹
    double tracking_th = 0.05;
    for (size_t i = 0; i < footsteps.size(); i++)
    {
        if (i == 0)
        {
            abs(polynomial(polynomial_param, (footstep_initial.x + footsteps.at(0).x)/2) - (footstep_initial.y + footsteps.at(0).y)/2) <= tracking_th;
        }
        else
        {
            abs(polynomial(polynomial_param, (footsteps.at(i - 1).x + footsteps.at(i).x)/2) - (footsteps.at(i - 1).y + footsteps.at(i).y)/2) <= tracking_th;
        }
    }
    

    matplotlibcpp::figure_size(1200, 780); // 设置图像大小
    matplotlibcpp::plot(x_data, y_data, {{"label", "P(x) = 3x^5 - 2x^4 + 5x^3 - x^2 + 7x - 4"}});
    matplotlibcpp::title("Five-degree Polynomial (matplotlib-cpp)");
    matplotlibcpp::xlabel("x");
    matplotlibcpp::ylabel("P(x)");
    matplotlibcpp::grid(true);
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}