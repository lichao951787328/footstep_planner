#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // for std::max
#include <glog/logging.h>
#include <matplotlibcpp.h>
using namespace std;
namespace plt = matplotlibcpp;
// ==========================================
// 辅助函数区域 (保持不变，仅修正 polynomial 类型支持)
// ==========================================

// 允许传入 CasADi MX 或 double 的多项式函数
template <typename T>
T polynomial_func(const Eigen::Vector<double, 6>& param, T x) {
    // CasADi 中需要使用 casadi::MX::pow 或重载的 pow，这里利用模板自动适配
    // 注意：如果是 MX，std::pow 可能报错，稳妥起见手写或用重载
    return param(0) * pow(x, 5) + param(1) * pow(x, 4) + param(2) * pow(x, 3) + 
           param(3) * pow(x, 2) + param(4) * x + param(5);
}

// 特化：针对 CasADi MX 的 pow
casadi::MX pow(casadi::MX x, int n) { return casadi::MX::pow(x, n); }

Eigen::Matrix4d get_autonomous_alip_matrix_A(double H_com, double mass, double g) {
    Eigen::Matrix4d A_c_autonomous;
    A_c_autonomous << 0, 0, 0, 1 / (mass * H_com),
                      0, 0, -1 / (mass * H_com), 0,
                      0, -mass * g, 0, 0,
                      mass * g, 0, 0, 0;
    return A_c_autonomous;
}

std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 2>> get_alip_matrices_with_input(double H_com, double mass, double g, double T_ss_dt) {
    Eigen::Matrix4d A_c = get_autonomous_alip_matrix_A(H_com, mass, g);
    Eigen::Matrix<double, 4, 2> B_c;
    B_c << 0, 0, 0, 0, 1, 0, 0, 1; // 简化力矩输入矩阵

    Eigen::Matrix4d A_d = (A_c * T_ss_dt).exp();
    Eigen::Matrix<double, 4, 2> B_d;
    
    // 简单处理奇异性
    if (std::abs(A_c.determinant()) > 1e-9) {
         B_d = A_c.inverse() * (A_d - Eigen::Matrix4d::Identity()) * B_c;
    } else {
         B_d = B_c * T_ss_dt; 
    }
    return {A_d, B_d};
}

std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 3>> get_alip_reset_map_matrices_detailed(double T_ds, double H_com, double mass, double g) {
    Eigen::Matrix4d A_c = get_autonomous_alip_matrix_A(H_com, mass, g);
    Eigen::Matrix4d Ar_ds = (A_c * T_ds).exp();

    // 简化 Reset Map B矩阵，主要处理坐标系切换: X_new = X_old - (P_new - P_old)
    // 假设双支撑期动力学影响包含在 Ar_ds 中
    Eigen::Matrix<double, 4, 3> P_map; 
    P_map.setZero();
    P_map(0,0) = 1.0; P_map(1,1) = 1.0; // 仅 x, y 位置发生坐标变换

    // B_r = -Ar_ds * P_map (对应公式 Ar * X - Ar * DeltaP)
    // 这里的推导依据您的原始代码逻辑保留，但确保维度匹配
    Eigen::Matrix<double, 4, 3> B_r = -Ar_ds * P_map;
    
    return {Ar_ds, B_r};
}

// ==========================================
// 主函数
// ==========================================

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]); 
    google::InstallFailureSignalHandler();
    // google::SetCommandLineOptionWithMode("FLAGS_minloglevel", "2");
    FLAGS_minloglevel = 0;
    FLAGS_colorlogtostderr = true;
    FLAGS_alsologtostderr = true;

    // 1. 轨迹与步数计算 (完全保留您的代码逻辑)
    Eigen::Vector<double, 6> polynomial_param;
    polynomial_param << 0, 0, -0.015625, 0.09375, 0, 0;

    double x_goal = 4.0;
    double y_goal = polynomial_func(polynomial_param, x_goal);
    LOG(INFO)<< "Goal position: (" << x_goal << ", " << y_goal << ")";
    double delta_x = 0.01;
    double delta_y = polynomial_func(polynomial_param, x_goal + delta_x) - polynomial_func(polynomial_param, x_goal);
    double yaw_goal = atan2(delta_y, delta_x);
    Eigen::Vector4d x_com_goal(x_goal, y_goal, 0, 0);

    size_t steps_1 = std::ceil(std::sqrt(x_goal * x_goal + y_goal * y_goal) / (0.28)) + 2; 
    LOG(INFO)<< "Determined number of steps based on distance: " << steps_1;
    double start_x = 0.0; // 修正: 假设从 0 开始
    double total_yaw_change = 0.0;
    double sample_step = 0.1;
    double prev_yaw = 0.0; 
    
    for (double x = start_x + sample_step; x <= x_goal; x += sample_step) {
        double y = polynomial_func(polynomial_param, x);
        double y_next = polynomial_func(polynomial_param, x + delta_x);
        double current_yaw = atan2(y_next - y, delta_x);
        
        double yaw_diff = current_yaw - prev_yaw;
        while (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
        while (yaw_diff < -M_PI) yaw_diff += 2 * M_PI;
        
        total_yaw_change += std::abs(yaw_diff);
        prev_yaw = current_yaw;
    }
    
    size_t steps_2 = std::ceil(total_yaw_change / (M_PI / 12)); 
    LOG(INFO)<< "Determined number of steps based on yaw change: " << steps_2;
    size_t N = std::max(steps_1, steps_2) ;
    LOG(INFO)<< "Determined number of steps N: " << N;

    if (N > 20)
    {
        LOG(WARNING)<< "Planned steps N is large (" << N << "), may lead to high computation time.";
    }
    
    // 2. 机器人参数与矩阵初始化，根据左右脚来决定的初始支撑脚
    bool initial_left_support = true;
    Eigen::Vector3d p_start_support_foot;
    if (initial_left_support)
    {
        p_start_support_foot = Eigen::Vector3d(0.0, 0.1, 0.0); // 初始左脚支撑
    }
    else
    {
        p_start_support_foot = Eigen::Vector3d(0.0, -0.1, 0.0); // 初始右脚支撑
    }
    
    // 先不考虑动力学
    // size_t k = 10; 
    // double H_com = 0.9, mass = 60, g = 9.81;
    // double swing_t = 0.75, double_support = 0.15;
    // double T_ss_dt = swing_t / k;
    
    // auto [A_d_mpc, B_d_mpc_vec] = get_alip_matrices_with_input(H_com, mass, g, T_ss_dt);
    // auto [Ar_reset, Br_reset_delta_p] = get_alip_reset_map_matrices_detailed(double_support, H_com, mass, g);
    
    // Eigen 转 CasADi DM
    // casadi::DM A_d_mpc_dm = casadi::DM::zeros(4, 4);
    // casadi::DM B_d_mpc_dm = casadi::DM::zeros(4, 2);
    // casadi::DM Ar_reset_dm = casadi::DM::zeros(4, 4);
    // casadi::DM Br_reset_dm = casadi::DM::zeros(4, 3);
    
    // 手动填充 DM (保留您的做法)
    // for (int i = 0; i < 4; ++i) {
        // for (int j = 0; j < 4; ++j) A_d_mpc_dm(i, j) = A_d_mpc(i, j);
        // for (int j = 0; j < 2; ++j) B_d_mpc_dm(i, j) = B_d_mpc_vec(i, j);
        // for (int j = 0; j < 4; ++j) Ar_reset_dm(i, j) = Ar_reset(i, j);
        // for (int j = 0; j < 3; ++j) Br_reset_dm(i, j) = Br_reset_delta_p(i, j);
    // }
    
    // 3. 优化器设置
    casadi::Opti opti = casadi::Opti();

    // --- 变量定义 (顺序不变) ---
    // casadi::MX X = opti.variable(4, N * (k + 1)); // 包含了初始状态
    // casadi::MX X_goal = opti.variable(4); 
    // casadi::MX U = opti.variable(2, N * k); 
    casadi::MX P = opti.variable(3, N); 

    // --- 代价函数 (顺序不变) ---
    casadi::MX J = 0;
    
    // 最小化输入
    // for (size_t i = 0; i < N * k; ++i) {
    //     J += U(0, i) * U(0, i) + U(1, i) * U(1, i);
    // }

    // 终点约束 Cost
    // double lambda_goal_yaw = 10000.0;
    
    // [修正]: ALIP 的 X 是局部状态。全局位置 = P_final + X_final_local
    // 我们用最后一步的 P 和 X_goal 来计算全局位置
    // casadi::MX P_final = P(casadi::Slice(), N - 1);
    
    // 全局位置误差
    // J += lambda_goal_yaw * ((X_goal(0) + P_final(0) - x_com_goal(0)) * (X_goal(0) + P_final(0) - x_com_goal(0)) + (X_goal(1) + P_final(1) - x_com_goal(1)) * (X_goal(1) + P_final(1) - x_com_goal(1)));
    
    // 终点 Yaw
    // J += lambda_goal_yaw * (P(2, N - 1) - yaw_goal) * (P(2, N - 1) - yaw_goal);

    // 转角均匀 Cost
    double lambda_yaw_smooth = 30.0;
    for (size_t i = 0; i < N - 1; ++i) {
        J += lambda_yaw_smooth * (P(2, i + 1) - P(2, i)) * (P(2, i + 1) - P(2, i));
    }

    // 步态均匀性
    double lambda_step_smooth = 15.0;
    for (size_t i = 0; i < N - 1; i++)
    {
        casadi::MX P_current = P(casadi::Slice(), i);
        casadi::MX P_next = P(casadi::Slice(), i + 1);
        J += lambda_step_smooth * ((P_next(0) - P_current(0)) * (P_next(0) - P_current(0)) + (P_next(1) - P_current(1)) * (P_next(1) - P_current(1)));
    }
    
    double lambda_yaw_guide = 300.0; // 权重给大一点
    for (size_t i = 0; i < N; ++i) {
        casadi::MX cx = P(0, i);
        // 计算当前 x 处的切线斜率
        casadi::MX dy_dx = 5 * polynomial_param(0) * casadi::MX::pow(cx, 4) + 
                        4 * polynomial_param(1) * casadi::MX::pow(cx, 3) + 
                        3 * polynomial_param(2) * casadi::MX::pow(cx, 2) + 
                        2 * polynomial_param(3) * cx + 
                        polynomial_param(4); 
        casadi::MX target_yaw = casadi::MX::atan(dy_dx);
        // 最小化当前 Yaw 与切线角的误差
        J += lambda_yaw_guide * (1.0 - casadi::MX::cos(P(2, i) - target_yaw));
    }

    double lambda_path_tracking = 1000.0; // 权重
    for (size_t i = 1; i < N; ++i) {
        casadi::MX mid_x = (P(0, i - 1) + P(0, i)) / 2.0;
        casadi::MX mid_y = (P(1, i - 1) + P(1, i)) / 2.0;
        
        // 计算理论 y
        casadi::MX poly_y = polynomial_func(polynomial_param, mid_x);
        
        // 最小化误差平方
        J += lambda_path_tracking * casadi::MX::sumsqr(poly_y - mid_y);
    }

    opti.minimize(J);   

    // --- 约束定义 (顺序不变) ---
    // 1. ALIP 动力学与 Reset Map (逻辑保持不变，仅修正矩阵乘法语法)
    // for (size_t i = 0; i < N; i++)
    // {
    //     for (size_t j = 0; j < k; j++)
    //     {
    //         size_t current_idx = i * (k + 1) + j;
    //         size_t next_idx = i * (k + 1) + j + 1;
    //         size_t control_idx = i * k + j;
            
    //         casadi::MX X_current = X(casadi::Slice(), current_idx);
    //         casadi::MX X_next = X(casadi::Slice(), next_idx);
    //         casadi::MX U_current = U(casadi::Slice(), control_idx);   
    //         // 使用 casadi::MX::mtimes 进行矩阵乘法
    //         opti.subject_to(X_next == casadi::MX::mtimes(A_d_mpc_dm, X_current) + casadi::MX::mtimes(B_d_mpc_dm, U_current));
    //     }
    //     // Reset 阶段
    //     if (i < N) // 统一处理，包括最后一个
    //     {
    //         size_t current_idx = i * (k + 1) + k; // 单脚支撑末态
    //         casadi::MX X_current = X(casadi::Slice(), current_idx);  
    //         casadi::MX X_next_state;
    //         if (i < N - 1) {
    //             size_t next_idx = (i + 1) * (k + 1);
    //             X_next_state = X(casadi::Slice(), next_idx);
    //         } else {
    //             X_next_state = X_goal; // 最后一个连接到 X_goal
    //         }
    //         casadi::MX P_current; // 这里指“迈出去之前的支撑脚”
    //         if (i == 0) {
    //             P_current = casadi::MX::vertcat({p_start_support_foot(0), p_start_support_foot(1), 0});
    //         } else {
    //             P_current = casadi::MX::vertcat({P(0, i - 1), P(1, i - 1), 0}); // 假设 yaw 影响忽略或 P(2)
    //         }
    //         casadi::MX P_next_step = casadi::MX::vertcat({P(0, i), P(1, i), 0}); 
    //         // X_next = Ar * X_curr + Br * (P_new - P_old)
    //         // 注意：Reset 时，P_new 是刚才迈出去落地的脚 P(i)，P_old 是刚才的支撑脚
    //         opti.subject_to(X_next_state == casadi::MX::mtimes(Ar_reset_dm, X_current) + casadi::MX::mtimes(Br_reset_dm, P_next_step - P_current));
    //     }
    // }
    
    // 2. 初始状态约束
    // [修正]: ALIP 模型中，X 是相对于支撑脚的局部状态。
    // 如果初始时刻机器人在原点 (0,0)，初始支撑脚在 (0, 0.1)，那么初始局部位置应该是 (0, -0.1)
    // 否则机器人第一步就会因为没有倾倒力矩而没法动
    // casadi::MX X_initial = X(casadi::Slice(), 0);
    // opti.subject_to(X_initial(0) == 0); 
    // 这一点是根据初始支撑脚是左右脚来决定的
    /**
     * @brief 这个初始状态非常重要，因为 ALIP 模型的状态是相对于支撑脚的位置。尤其是角动量部分。
     * 
     */
    // opti.subject_to(X_initial(1) == -p_start_support_foot(1)); // 相对位置 y = 0 - 0.1 = -0.1
    // opti.subject_to(X_initial(2) == 0); 
    // opti.subject_to(X_initial(3) == 0); 

    // 初始落脚点约束
    casadi::MX P_initial = P(casadi::Slice(), 0);
    opti.subject_to(P_initial(0) == p_start_support_foot(0));
    opti.subject_to(P_initial(1) == p_start_support_foot(1));
    opti.subject_to(P_initial(2) == 0); // 假设初始朝向为0

    // 终点约束
    // 新的终点约束方案：中点重合 + 连线垂直
    // ==========================================
    // 1. 获取最后两步的状态变量
    casadi::MX P_final = P(casadi::Slice(), N - 1); // 最后一步落脚点
    casadi::MX P_prev  = P(casadi::Slice(), N - 2); // 倒数第二步落脚点

    // 2. 约束两脚的朝向都必须对齐目标 Yaw
    //    (虽然是连线垂直，但脚本身的朝向还是应该朝前的)
    opti.subject_to(P_final(2) == yaw_goal);
    opti.subject_to(P_prev(2)  == yaw_goal);

    // 3. 约束中点位置 (Midpoint Constraint)
    //    (P_final + P_prev) / 2 = Goal
    opti.subject_to( (P_final(0) + P_prev(0)) / 2.0 == x_goal );
    opti.subject_to( (P_final(1) + P_prev(1)) / 2.0 == y_goal );

    // 4. 约束连线方向垂直于目标朝向 (Perpendicular Constraint)
    //    原理：向量 Dot( (P_final - P_prev), Goal_Heading_Vector ) == 0
    //    Goal_Heading_Vector = [cos(yaw_goal), sin(yaw_goal)]
    
    // 预计算目标 Yaw 的三角函数（注意：这里用 double 常量即可，不需要 MX，因为 yaw_goal 是已知常数）
    double cg = std::cos(yaw_goal);
    double sg = std::sin(yaw_goal);

    casadi::MX dx_feet = P_final(0) - P_prev(0);
    casadi::MX dy_feet = P_final(1) - P_prev(1);

    // 点积为 0 即为垂直
    opti.subject_to(dx_feet * cg + dy_feet * sg == 0 );

    // 5. (可选但推荐) 最小站立宽度约束
    //    防止求解器为了偷懒，把两只脚重叠在一起放在终点 (虽然物理上不可能，但数学上是个可行解)
    double min_stance_width = 0.15; // 最小允许两脚间距 15cm
    casadi::MX dist_sq_feet = dx_feet * dx_feet + dy_feet * dy_feet;
    opti.subject_to( dist_sq_feet >= min_stance_width * min_stance_width );
    
    // 最大宽度约束通常由双圆运动学约束涵盖了，但为了保险也可以加一个
    double max_stance_width = 0.40; 
    opti.subject_to( dist_sq_feet <= max_stance_width * max_stance_width );

    // 3. 力矩约束
    // double max_roll = 100.0; // 稍微放宽一点，避免无解
    // double max_pitch = 100.0;
    // for (size_t i = 0; i < N * k; ++i)
    // {
    //     casadi::MX U_current = U(casadi::Slice(), i);
    //     // 使用 norm_2

    //     double limit_sq = max_roll * max_roll + max_pitch * max_pitch;
    //     // 使用 sumsqr (u_x^2 + u_y^2) 替代 norm_2
    //     opti.subject_to( U_current(0)*U_current(0) + U_current(1)*U_current(1) <= limit_sq );
    // }

    // 4. 相邻步转角约束
    double max_step_yaw = M_PI / 12; // 稍微放宽
    for (size_t i = 0; i < N - 1; ++i)
    {
        casadi::MX P_current = P(casadi::Slice(), i);
        casadi::MX P_next = P(casadi::Slice(), i + 1);
        opti.subject_to(casadi::MX::abs(P_next(2) - P_current(2)) <= max_step_yaw);
    }

    // 5. 运动学约束 (您要求的方案：双圆约束)
    for (size_t i = 1; i < N; ++i) // 此编号为摆动脚的编号
    {
        bool cuurent_left_support;
        // 如果摆动脚是偶数步，那么支撑脚是奇数步，也就是支撑脚是右脚
        if (i % 2 == 0) cuurent_left_support = !initial_left_support; 
        else cuurent_left_support = initial_left_support;

        casadi::MX support_center = casadi::MX::vertcat({P(0, i - 1), P(1, i - 1), 0});
        casadi::MX yaw_support = P(2, i - 1);
        
        double deta1 = 1.8; // 内圆圆心偏移量
        double deta2 = 0.35; // 外圆圆心偏移量
        double dis_th1 = 1.68; // 内圆半径
        double dis_th2 = 0.675; // 外圆半径

        casadi::MX P_next_pos = P(casadi::Slice(), i);
        casadi::MX sin_yaw = casadi::MX::sin(yaw_support);
        casadi::MX cos_yaw = casadi::MX::cos(yaw_support);

        // 逻辑保持您的代码不变，仅修正变量类型
        if (!cuurent_left_support) { // Right leg support

            // inner circle
            casadi::MX p1_x = support_center(0) - (deta1) * sin_yaw;
            casadi::MX p1_y = support_center(1) + (deta1) * cos_yaw;
            casadi::MX dist1_sq = casadi::MX::pow(P_next_pos(0) - p1_x, 2) + casadi::MX::pow(P_next_pos(1) - p1_y, 2);
            opti.subject_to(dist1_sq <= dis_th1 * dis_th1);
            
            // outer circle
            casadi::MX p2_x = support_center(0) - (-deta2) * sin_yaw;
            casadi::MX p2_y = support_center(1) + (-deta2) * cos_yaw;
            casadi::MX dist2_sq = casadi::MX::pow(P_next_pos(0) - p2_x, 2) + casadi::MX::pow(P_next_pos(1) - p2_y, 2);
            opti.subject_to(dist2_sq <= dis_th2 * dis_th2);
            
        } else { // Left leg support
            // inner circle
            casadi::MX p1_x = support_center(0) - (-deta1) * sin_yaw;
            casadi::MX p1_y = support_center(1) + (-deta1) * cos_yaw;
            casadi::MX dist1_sq = casadi::MX::pow(P_next_pos(0) - p1_x, 2) + casadi::MX::pow(P_next_pos(1) - p1_y, 2);
            opti.subject_to(dist1_sq <= dis_th1 * dis_th1);
            
            // outer circle
            casadi::MX p2_x = support_center(0) - (deta2) * sin_yaw;
            casadi::MX p2_y = support_center(1) + (deta2) * cos_yaw;
            casadi::MX dist2_sq = casadi::MX::pow(P_next_pos(0) - p2_x, 2) + casadi::MX::pow(P_next_pos(1) - p2_y, 2);
            opti.subject_to(dist2_sq <= dis_th2 * dis_th2);
        }
    }   

    // 6. 路径跟随约束 (修改点：按照您的要求，约束两相邻落脚点的中点在曲线上)
    // double tracking_th = 0.05;
    // for (size_t i = 1; i < N; ++i)
    // {
    //     casadi::MX mid_x;
    //     casadi::MX mid_y;
    //     // 计算相邻落脚点的中点
    //     // P(i) 与 P(i-1) 的中点
    //     mid_x = (P(0, i - 1) + P(0, i)) / 2.0;
    //     mid_y = (P(1, i - 1) + P(1, i)) / 2.0;

    //     // 计算该 mid_x 在多项式曲线上对应的理论 y 值
    //     // 注意：传入 poly 函数的参数必须是 casadi::MX
    //     casadi::MX poly_y = polynomial_func(polynomial_param, mid_x);

    //     // 约束：中点的实际 y 与 理论 y 接近
    //     opti.subject_to(casadi::MX::abs(poly_y - mid_y) <= tracking_th);
    // }
    
    // --- 求解 ---
    opti.solver("ipopt");

    // --- 关键修复：添加初始猜测 (Warm Start) ---
    for (int i = 0; i < N - 1; ++i) {
        // 猜测落脚点沿着 x 轴匀速前进，每步走 0.25 米
        double guess_px = p_start_support_foot(0) + (i) * 0.25;
        
        // 根据x计算y,根据xy计算yaw
        double guess_py = polynomial_func(polynomial_param, guess_px);
        double delta_x = 0.01;
        double delta_y = polynomial_func(polynomial_param, guess_px + delta_x) - polynomial_func(polynomial_param, guess_px);
        double guess_yaw = atan2(delta_y, delta_x);

        // 根据xy及yaw,确定此步如果是左脚还是右脚,并调整y值
        bool cuurent_left_support;
        if (i % 2 == 0) cuurent_left_support = !initial_left_support; 
        else cuurent_left_support = initial_left_support;
        double foot_offset = 0.1; // 假设双脚间距为0.2米
        if (cuurent_left_support) {
            // 左脚
            // guess_py += foot_offset;
            guess_px -= foot_offset * sin(guess_yaw);
            guess_py += foot_offset * cos(guess_yaw);
        } else {
            // 右脚 
            guess_px += foot_offset * sin(guess_yaw);
            guess_py -= foot_offset * cos(guess_yaw);
        }   
        // 设置初始猜测值
        opti.set_initial(P(0, i), guess_px);
        opti.set_initial(P(1, i), guess_py);
        opti.set_initial(P(2, i), guess_yaw); 
    }

    // 最后一步的猜测
    opti.set_initial(P(0, N - 1), x_goal);
    opti.set_initial(P(1, N - 1), y_goal);
    opti.set_initial(P(2, N - 1), yaw_goal);

    // 

    // // 猜测力矩为 0 (被动行走)
    // opti.set_initial(U, 0.0);

    std::vector<double> res_px, res_py, res_yaw;

    try {
        casadi::OptiSol sol = opti.solve();
        std::cout << "Optimization Success!" << std::endl;
        
        // 简单打印结果验证
        res_px = std::vector<double>(sol.value(P(0, casadi::Slice())));
        res_py = std::vector<double>(sol.value(P(1, casadi::Slice())));
        res_yaw = std::vector<double>(sol.value(P(2, casadi::Slice())));
        for(size_t i=0; i<res_px.size(); ++i) {
            std::cout << "Step " << i << ": (" << res_px[i] << ", " << res_py[i] << ", " <<res_yaw[i] << ")" << std::endl;
        }

    } catch(std::exception& e) {
        std::cerr << "Optimization Failed: " << e.what() << std::endl;
        // opti.debug().value(P);
    }

    /**
     * 画曲线图使用的参数
     */
    // 获取结果数据
        
    // --- 可视化部分 ---
    
    // 1. 准备引导曲线数据
    std::vector<double> ref_x, ref_y;
    for(double x = 0; x <= x_goal + 0.5; x += 0.05) {
        ref_x.push_back(x);
        ref_y.push_back(polynomial_func(polynomial_param, x));
    }

    // 2. 准备落脚点数据
    // 2. 轨迹连线数据
    std::vector<double> traj_x, traj_y;
    traj_x.push_back(p_start_support_foot(0));
    traj_y.push_back(p_start_support_foot(1));
    for(size_t i=0; i<res_px.size(); ++i) {
        traj_x.push_back(res_px[i]);
        traj_y.push_back(res_py[i]);
    }

    // 3. 脚印分类与箭头数据准备
    std::vector<double> left_foot_x, left_foot_y, left_u, left_v;
    std::vector<double> right_foot_x, right_foot_y, right_u, right_v;
    
    double arrow_len = 0.1; // 箭头长度

    // 处理起始脚 (Left)
    left_foot_x.push_back(p_start_support_foot(0));
    left_foot_y.push_back(p_start_support_foot(1));
    left_u.push_back(arrow_len * cos(p_start_support_foot(2)));
    left_v.push_back(arrow_len * sin(p_start_support_foot(2)));

    for(size_t i=0; i<res_px.size(); ++i) {
        double theta = res_yaw[i];
        double u_comp = arrow_len * cos(theta);
        double v_comp = arrow_len * sin(theta);

        // 如果初始是 Left，那么 i=0 (Step 1) 是 Right，i=1 是 Left...
        if (i % 2 == 0) { // Right Foot
            right_foot_x.push_back(res_px[i]);
            right_foot_y.push_back(res_py[i]);
            right_u.push_back(u_comp);
            right_v.push_back(v_comp);
        } else { // Left Foot
            left_foot_x.push_back(res_px[i]);
            left_foot_y.push_back(res_py[i]);
            left_u.push_back(u_comp);
            left_v.push_back(v_comp);
        }
    }
    // 3. 开始绘图
    plt::figure_size(1200, 800); // 设置窗口大小
    
    // 画引导线 (虚线，灰色)
    plt::plot(ref_x, ref_y, "k--"); 
    
    // 画轨迹连线 (灰色细线)
    plt::plot(traj_x, traj_y, "gray");
    
    // 左脚：红色圆点 + 红色箭头
    plt::scatter(left_foot_x, left_foot_y, 50.0, {{"color", "red"}, {"label", "Left Foot"}});
    plt::quiver(left_foot_x, left_foot_y, left_u, left_v, {{"color", "red"}});

    // 右脚：蓝色圆点 + 蓝色箭头
    plt::scatter(right_foot_x, right_foot_y, 50.0, {{"color", "blue"}, {"label", "Right Foot"}});
    plt::quiver(right_foot_x, right_foot_y, right_u, right_v, {{"color", "blue"}});
    
    // 标记起点和终点
    // std::vector<double> start_x = {p_start_support_foot(0)};
    // std::vector<double> start_y = {p_start_support_foot(1)};
    // plt::text(start_x[0], start_y[0], "Start");
    
    std::vector<double> goal_pt_x = {x_goal};
    std::vector<double> goal_pt_y = {y_goal};
    plt::scatter(goal_pt_x, goal_pt_y, 100.0, {{"color", "green"}, {"marker", "*"}, {"label", "Goal"}});

    // 设置图形属性
    plt::title("Footstep Planning Result with ALIP");
    plt::xlabel("X [m]");
    plt::ylabel("Y [m]");
    plt::axis("equal"); // 保证 XY 比例一致，不然看着是歪的
    plt::legend();      // 显示图例
    plt::grid(true);    // 显示网格
    
    // 保存并显示
    plt::save("footstep_plan.png");
    plt::show();

    return 0;
}