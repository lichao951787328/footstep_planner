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
#include <chrono>

#define ALIP

#define GOAL_Hard_Constraint
#define GOAL_Hard_Constraint_alip
using namespace std;
namespace plt = matplotlibcpp;
// ==========================================
// 辅助函数区域 (保持不变，仅修正 polynomial 类型支持)。动力学模型部分见2023,2022年，alip_feetstep_planning的论文，其他部分见2014年的论文。
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


// 机器人动力学的矩阵计算函数（基础矩阵）
Eigen::Matrix4d get_autonomous_alip_matrix_A(double H_com, double mass, double g) {
    Eigen::Matrix4d A_c_autonomous;
    A_c_autonomous << 0, 0, 0, 1 / (mass * H_com),
                      0, 0, -1 / (mass * H_com), 0,
                      0, -mass * g, 0, 0,
                      mass * g, 0, 0, 0;
    return A_c_autonomous;
}

// 单脚支撑期状态转移函数矩阵（含输入矩阵）
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

// 双脚支撑期状态转移矩阵及 Reset Map 矩阵（含位置变换）
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

    Eigen::Matrix<double, 4, 3> B_cop;
    B_cop <<0,0,0,
            0,0,0,
            0,mass * g,0,
            -mass * g,0,0;
    Eigen::Matrix<double, 4, 3> B_ds = Ar_ds * A_c.inverse() * (A_c.inverse() * (Eigen::Matrix4d::Identity() - Ar_ds.inverse())/T_ds - Ar_ds.inverse()) * B_cop;
    Eigen::Matrix<double, 4, 3> B_r = B_ds + P_map;
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

    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. 轨迹与步数计算 (完全保留您的代码逻辑)
    Eigen::Vector<double, 6> polynomial_param;
    polynomial_param << 0, 0, -0.015625, 0.09375, 0, 0;

    // 设定起点和终点
    double x_goal = 4.0;
    double y_goal = polynomial_func(polynomial_param, x_goal);
    LOG(INFO)<< "Goal position: (" << x_goal << ", " << y_goal << ")";
    double delta_x = 0.01;
    double delta_y = polynomial_func(polynomial_param, x_goal + delta_x) - polynomial_func(polynomial_param, x_goal);
    double yaw_goal = atan2(delta_y, delta_x);
    // double yaw_goal = 45 / 180.0 * M_PI; // 目标朝向设为-30度

    // 定义目标质心状态
    Eigen::Vector4d x_com_goal(x_goal, y_goal, 0, 0);

    // 2. 机器人参数与矩阵初始化，根据左右脚来决定的初始支撑脚
    // 假设机器人初始时刻左脚支撑，并据此确定机器人的左脚位置。初始时刻，机器人base在地面上的投影点为坐标远点，往前x,往左y,向上z。
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

    // 估计到达终点的距离与步数
    LOG(INFO) << "line distance: " << std::sqrt(std::pow(x_goal - p_start_support_foot(0), 2) + std::pow(y_goal - p_start_support_foot(1), 2));
    // props.total_time 保持不变 (由 N 决定)
    
    // 这个还需要改进，考虑路径曲率
    size_t steps_1 = std::ceil((std::sqrt(std::pow(x_goal - p_start_support_foot(0), 2) + std::pow(y_goal - p_start_support_foot(1), 2))) / (0.3)) + 2; 
    // size_t steps_1 = std::ceil(props.total_dist / (0.3)) + 2; 
    LOG(INFO)<< "Determined number of steps based on distance: " << steps_1;

    // 估计到达终点的转角变化量与步数
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

    size_t steps_2 = std::ceil(total_yaw_change / (M_PI / 12)) + 2; 
    LOG(INFO)<< "Determined number of steps based on yaw change: " << steps_2;
    size_t N = std::max(steps_1, steps_2) ;
    LOG(INFO)<< "Determined number of steps N: " << N;

    if (N > 20)
    {
        LOG(WARNING)<< "Planned steps N is large (" << N << "), may lead to high computation time.";
    }


    // 3. 优化器设置
    casadi::Opti opti = casadi::Opti();

    // --- 变量定义 (顺序不变) ---
    // 定义优化变量，即落脚点，每个落脚点包含 (x, y, yaw)。暂时不考虑 z 高度变化。
    casadi::MX P = opti.variable(3, N); 

    // --- 代价函数 (顺序不变) ---
    casadi::MX J = 0;


    // 转角均匀 Cost
    double lambda_yaw_smooth = 300.0;
    for (size_t i = 0; i < N - 1; ++i) {
        J += lambda_yaw_smooth * (P(2, i + 1) - P(2, i)) * (P(2, i + 1) - P(2, i));
    }

    // 步态均匀性
    double lambda_step_smooth = 30.0; // 这个不作强制性约束，因为在引入动力学后这一项可由动力学保证
    for (size_t i = 0; i < N - 1; i++)
    {
        casadi::MX P_current = P(casadi::Slice(), i);
        casadi::MX P_next = P(casadi::Slice(), i + 1);
        J += lambda_step_smooth * ((P_next(0) - P_current(0)) * (P_next(0) - P_current(0)) + (P_next(1) - P_current(1)) * (P_next(1) - P_current(1)));
    }

    // 转向角跟随局部path Cost
    double lambda_yaw_guide = 5.0; // 权重给大一点
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

    // 避免出现螃蟹步
    double lambda_crab_avoid = 80.0;
    for (size_t i = 0; i < N - 2; ++i) {
        // 首先确定当前是左脚还是右脚，再选择后一次对应的左右脚，再计算两个落脚点的方向，该方向与当前朝向的夹角
        casadi::MX delta_x = P(0, i + 2) - P(0, i);
        casadi::MX delta_y = P(1, i + 2) - P(1, i);
        casadi::MX yaw = P(2, i);
        // 当前是左脚落脚点，下一次也是左脚落脚点
        casadi::MX lateral_disp = -casadi::MX::sin(yaw) * delta_x + casadi::MX::cos(yaw) * delta_y;
        J += lambda_crab_avoid * (lateral_disp * lateral_disp);
        // J += lambda_crab_avoid * (1.0 - casadi::MX::cos(step_yaw - P(2, current_foot_index)));
    }

    // 路径跟踪 Cost。规划的落脚点要跟踪多项式path
    double lambda_path_tracking = 2.0; // 权重
    for (size_t i = 1; i < N; ++i) {
        casadi::MX mid_x = (P(0, i - 1) + P(0, i)) / 2.0;
        casadi::MX mid_y = (P(1, i - 1) + P(1, i)) / 2.0;
        
        // 计算理论 y
        casadi::MX poly_y = polynomial_func(polynomial_param, mid_x);
        
        // 最小化误差平方
        J += lambda_path_tracking * casadi::MX::sumsqr(poly_y - mid_y);
    }

    // 终点方向角跟踪 Cost
    double lambda_goal_yaw_tracking = 500.0;
    J += lambda_goal_yaw_tracking * (1.0 - casadi::MX::cos(P(2, N - 1) - yaw_goal));


    opti.minimize(J);

    // --- 约束定义 (顺序不变) ---
    // 1. ALIP 动力学与 Reset Map (逻辑保持不变，仅修正矩阵乘法语法)

    // 初始落脚点约束
    casadi::MX P_initial = P(casadi::Slice(), 0);
    opti.subject_to(P_initial(0) == p_start_support_foot(0));
    opti.subject_to(P_initial(1) == p_start_support_foot(1));
    opti.subject_to(P_initial(2) == 0); // 假设初始朝向为0
 
#ifdef GOAL_Hard_Constraint
    casadi::MX P_final = P(casadi::Slice(), N - 1); // 最后一步落脚点
    casadi::MX P_prev  = P(casadi::Slice(), N - 2); // 倒数第二步落脚点

    // 2. 约束两脚的朝向都必须对齐目标 Yaw
    //    (虽然是连线垂直，但脚本身的朝向还是应该朝前的)
    opti.subject_to(P_final(2) == yaw_goal);
    opti.subject_to(P_prev(2)  == yaw_goal);

    casadi::MX dx_feet = P_final(0) - P_prev(0);
    casadi::MX dy_feet = P_final(1) - P_prev(1);


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

    

    // 点积为 0 即为垂直
    opti.subject_to(dx_feet * cg + dy_feet * sg == 0 );

    // 5. (可选但推荐) 最小站立宽度约束
    // 防止求解器为了偷懒，把两只脚重叠在一起放在终点 (虽然物理上不可能，但数学上是个可行解)
    double min_stance_width = 0.15; // 最小允许两脚间距 15cm
    casadi::MX dist_sq_feet = dx_feet * dx_feet + dy_feet * dy_feet;
    opti.subject_to( dist_sq_feet >= min_stance_width * min_stance_width );
    
    // 最大宽度约束通常由双圆运动学约束涵盖了，但为了保险也可以加一个
    double max_stance_width = 0.30; 
    opti.subject_to( dist_sq_feet <= max_stance_width * max_stance_width );
#endif

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

    
    // --- 求解 ---
    opti.solver("ipopt");

    // 添加初始猜测 (Warm Start) ---
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



    std::vector<double> res_px, res_py, res_yaw;
    std::vector<Eigen::Vector2d> control_u;
    int time_cost = 0;
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
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Optimization Time: " << duration << " ms" << std::endl;
        time_cost = duration;

    } catch(std::exception& e) {
        std::cerr << "Optimization Failed: " << e.what() << std::endl;
        // opti.debug().value(P);
    }

    // 在这加第二层，第一层使用运动学和跟随实现落脚点规划，第二层使用 ALIP 优化力矩和微调落脚点

#ifdef ALIP

    auto start_time_alip = std::chrono::high_resolution_clock::now();
    // 动力学参数
    size_t k = 10; // 将单脚支撑期分成了k段，也就是从单脚支撑期开始到单脚脚支撑期结束共有k+1个时刻，其实每个步态也就是这k+1个时刻，因为双脚支撑的末尾正好对应下一次单脚支撑期开始状态。在这k+1个时刻中，只有k个时刻有控制输入。
    double H_com = 0.9, mass = 60, g = 9.81;
    double swing_t = 0.8, double_support = 0.2;
    double T_ss_dt = swing_t / k;

    auto [A_d_mpc, B_d_mpc_vec] = get_alip_matrices_with_input(H_com, mass, g, T_ss_dt);
    auto [Ar_reset, Br_reset_delta_p] = get_alip_reset_map_matrices_detailed(double_support, H_com, mass, g);
    
    // Eigen 转 CasADi DM
    casadi::DM A_d_mpc_dm = casadi::DM::zeros(4, 4);
    casadi::DM B_d_mpc_dm = casadi::DM::zeros(4, 2);
    casadi::DM Ar_reset_dm = casadi::DM::zeros(4, 4);
    casadi::DM Br_reset_dm = casadi::DM::zeros(4, 3);
    
    // 手动填充 DM 
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) A_d_mpc_dm(i, j) = A_d_mpc(i, j);
        for (int j = 0; j < 2; ++j) B_d_mpc_dm(i, j) = B_d_mpc_vec(i, j);
        for (int j = 0; j < 4; ++j) Ar_reset_dm(i, j) = Ar_reset(i, j);
        for (int j = 0; j < 3; ++j) Br_reset_dm(i, j) = Br_reset_delta_p(i, j);
    }
    casadi::Opti opti_alip= casadi::Opti();
    casadi::MX X = opti_alip.variable(4, N * (k + 1)); // 包含了初始状态，没有包含终点状态
    casadi::MX X_goal = opti_alip.variable(4); // 终点状态变量
    casadi::MX P_alip = opti_alip.variable(3, N); 
    // X_goal(2) = 0; X_goal(3) = 0; // 终点状态设为零
    casadi::MX U = opti_alip.variable(2, N * k); // 控制输入变量
    // 最小化输入

    casadi::MX J_alip = 0;

    // 转角均匀 Cost
    double lambda_yaw_smooth_alip = 300.0;
    for (size_t i = 0; i < N - 1; ++i) {
        J_alip += lambda_yaw_smooth_alip * (P_alip(2, i + 1) - P_alip(2, i)) * (P_alip(2, i + 1) - P_alip(2, i));
    }

    // 步态均匀性
    double lambda_step_smooth_alip = 13.0; // 这个不作强制性约束，因为在引入动力学后这一项可由动力学保证
    for (size_t i = 0; i < N - 1; i++)
    {
        casadi::MX P_current = P_alip(casadi::Slice(), i);
        casadi::MX P_next = P_alip(casadi::Slice(), i + 1);
        J_alip += lambda_step_smooth_alip * ((P_next(0) - P_current(0)) * (P_next(0) - P_current(0)) + (P_next(1) - P_current(1)) * (P_next(1) - P_current(1)));
    }

    // 转向角跟随 Cost
    // double lambda_yaw_guide = 5.0; // 权重给大一点
    // for (size_t i = 0; i < N; ++i) {
    //     casadi::MX cx = P(0, i);
    //     // 计算当前 x 处的切线斜率
    //     casadi::MX dy_dx = 5 * polynomial_param(0) * casadi::MX::pow(cx, 4) + 
    //                     4 * polynomial_param(1) * casadi::MX::pow(cx, 3) + 
    //                     3 * polynomial_param(2) * casadi::MX::pow(cx, 2) + 
    //                     2 * polynomial_param(3) * cx + 
    //                     polynomial_param(4); 
    //     casadi::MX target_yaw = casadi::MX::atan(dy_dx);
    //     // 最小化当前 Yaw 与切线角的误差
    //     J_alip += lambda_yaw_guide * (1.0 - casadi::MX::cos(P_alip(2, i) - target_yaw));
    // }

    // 避免出现螃蟹步
    double lambda_crab_avoid_alip = 80.0;
    for (size_t i = 0; i < N - 2; ++i) {
        // 首先确定当前是左脚还是右脚，再选择后一次对应的左右脚，再计算两个落脚点的方向，该方向与当前朝向的夹角
        casadi::MX delta_x = P_alip(0, i + 2) - P_alip(0, i);
        casadi::MX delta_y = P_alip(1, i + 2) - P_alip(1, i);
        casadi::MX yaw = P_alip(2, i);
        // 当前是左脚落脚点，下一次也是左脚落脚点
        casadi::MX lateral_disp = -casadi::MX::sin(yaw) * delta_x + casadi::MX::cos(yaw) * delta_y;
        J_alip += lambda_crab_avoid_alip * (lateral_disp * lateral_disp);
    }

    double lambda_u = 0.6;
    // 控制输入最小化 Cost，尽量避免主动使用踝关节力矩来维持平衡
    for (size_t i = 0; i < N * k; ++i) {
        J_alip += lambda_u * (U(0, i) * U(0, i) + U(1, i) * U(1, i));
    }

    // // 终点约束 Cost
    // // [修正]: ALIP 的 X 是局部状态。全局位置 = P_final + X_final_local
    // // 我们用最后一步的 P 和 X_goal 来计算全局位置
    double lambda_goal_yaw = 500.0;
    casadi::MX P_final_com = P_alip(casadi::Slice(), N - 1);
    J_alip += lambda_goal_yaw * ((X_goal(0) + P_final_com(0) - x_com_goal(0)) * (X_goal(0) + P_final_com(0) - x_com_goal(0)) + (X_goal(1) + P_final_com(1) - x_com_goal(1)) * (X_goal(1) + P_final_com(1) - x_com_goal(1)));

    double lambda_terminal_state = 1000.0; // 权重很大，效果接近硬约束
    
    // 惩罚末端位置偏差 (防止摔倒)
    // J_alip += lambda_terminal_state * (casadi::MX::pow(X_goal(0), 2) + casadi::MX::pow(X_goal(1), 2));
    
    // 惩罚末端动量 (刹车)
    J_alip += lambda_terminal_state * (casadi::MX::pow(X_goal(2), 2) + casadi::MX::pow(X_goal(3), 2));

    // 加一个代价，使加alip模型后的步态尽量跟随上层规划出来的落脚点
    double lambda_foot_tracking = 40.0;
    for (size_t i = 0; i < N; ++i) {
        double ref_x = res_px[i]; // 使用数值
        double ref_y = res_py[i];
        double ref_yaw = res_yaw[i];
        casadi::MX P_alip_current = P_alip(casadi::Slice(), i); // ALIP 优化的落脚点
        J_alip += lambda_foot_tracking * ((ref_x - P_alip_current(0)) * (ref_x - P_alip_current(0)) + (ref_y - P_alip_current(1)) * (ref_y - P_alip_current(1)));
        J_alip += lambda_foot_tracking * (1.0 - casadi::MX::cos(ref_yaw - P_alip_current(2)));
    }   

    opti_alip.minimize(J_alip);

    // 约束部分
    // 单脚、双脚支撑期的动力学约束
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < k; j++)// 单脚支撑期编号
        {
            size_t current_idx = i * (k + 1) + j;
            size_t next_idx = i * (k + 1) + j + 1;
            size_t control_idx = i * k + j;
            
            casadi::MX X_current = X(casadi::Slice(), current_idx);
            casadi::MX X_next = X(casadi::Slice(), next_idx);
            casadi::MX U_current = U(casadi::Slice(), control_idx);   
            // 使用 casadi::MX::mtimes 进行矩阵乘法
            opti_alip.subject_to(X_next == casadi::MX::mtimes(A_d_mpc_dm, X_current) + casadi::MX::mtimes(B_d_mpc_dm, U_current));
        }
        // Reset 阶段
        if (i < N) // 统一处理，包括最后一个
        {
            size_t current_idx = i * (k + 1) + k; // 单脚支撑末态
            casadi::MX X_current = X(casadi::Slice(), current_idx);  
            casadi::MX X_next_state;
            if (i < N - 1) {
                size_t next_idx = (i + 1) * (k + 1);
                X_next_state = X(casadi::Slice(), next_idx);
            } else {
                X_next_state = X_goal; // 最后一个连接到 X_goal
            }
            casadi::MX P_current; // 这里指“迈出去之前的支撑脚”
            if (i == 0) {
                P_current = casadi::MX::vertcat({p_start_support_foot(0), p_start_support_foot(1), 0});
            } else {
                P_current = casadi::MX::vertcat({P_alip(0, i - 1), P_alip(1, i - 1), 0}); // 假设 yaw 影响忽略或 P(2)
            }
            casadi::MX P_next_step = casadi::MX::vertcat({P_alip(0, i), P_alip(1, i), 0}); 
            // X_next = Ar * X_curr + Br * (P_new - P_old)
            // 注意：Reset 时，P_new 是刚才迈出去落地的脚 P(i)，P_old 是刚才的支撑脚
            opti_alip.subject_to(X_next_state == casadi::MX::mtimes(Ar_reset_dm, X_current) + casadi::MX::mtimes(Br_reset_dm, P_next_step - P_current));
        }
    }
    
    // 2. 初始状态约束
    casadi::MX X_initial = X(casadi::Slice(), 0);
    double v_start_expected = 0.3; // 0.3 m/s 的启动速度
    double Ly_start = mass * H_com * v_start_expected;

    opti_alip.subject_to(X_initial(0) == 0); 
    opti_alip.subject_to(X_initial(1) == -0.01); // 保持 y 轴相对位置
    opti_alip.subject_to(X_initial(2) == 0); // Lx 依然可以是 0 (假设还没开始侧摆)
    opti_alip.subject_to(X_initial(3) == 0.5); // <--- 给定初速度

    // 初始落脚点约束
    casadi::MX P_initial_alip = P_alip(casadi::Slice(), 0);
    opti_alip.subject_to(P_initial_alip(0) == p_start_support_foot(0));
    opti_alip.subject_to(P_initial_alip(1) == p_start_support_foot(1));
    opti_alip.subject_to(P_initial_alip(2) == 0); // 假设初始朝向为0

    double max_roll = 50.0; // 稍微放宽一点，避免无解
    double max_pitch = 50.0;
    for (size_t i = 0; i < N * k; ++i)
    {
        casadi::MX U_current = U(casadi::Slice(), i);
        // 使用 norm_2
        double limit_sq = max_roll * max_roll + max_pitch * max_pitch;
        // 使用 sumsqr (u_x^2 + u_y^2) 替代 norm_2
        opti_alip.subject_to( U_current(0)*U_current(0) + U_current(1)*U_current(1) <= limit_sq );
    }
    
#ifdef GOAL_Hard_Constraint_alip
    casadi::MX P_final_alip = P_alip(casadi::Slice(), N - 1); // 最后一步落脚点
    casadi::MX P_prev_alip  = P_alip(casadi::Slice(), N - 2); // 倒数第二步落脚点

    // 2. 约束两脚的朝向都必须对齐目标 Yaw
    //    (虽然是连线垂直，但脚本身的朝向还是应该朝前的)
    opti_alip.subject_to(P_final_alip(2) == yaw_goal);
    opti_alip.subject_to(P_prev_alip(2)  == yaw_goal);

    casadi::MX dx_feet_alip = P_final_alip(0) - P_prev_alip(0);
    casadi::MX dy_feet_alip = P_final_alip(1) - P_prev_alip(1);


    // 3. 约束中点位置 (Midpoint Constraint)
    //    (P_final + P_prev) / 2 = Goal
    opti_alip.subject_to( (P_final_alip(0) + P_prev_alip(0)) / 2.0 == x_goal );
    opti_alip.subject_to( (P_final_alip(1) + P_prev_alip(1)) / 2.0 == y_goal );

    // 4. 约束连线方向垂直于目标朝向 (Perpendicular Constraint)
    //    原理：向量 Dot( (P_final - P_prev), Goal_Heading_Vector ) == 0
    //    Goal_Heading_Vector = [cos(yaw_goal), sin(yaw_goal)]
    
    // 预计算目标 Yaw 的三角函数（注意：这里用 double 常量即可，不需要 MX，因为 yaw_goal 是已知常数）
    double cg_alip = std::cos(yaw_goal);
    double sg_alip = std::sin(yaw_goal);

    

    // 点积为 0 即为垂直
    opti_alip.subject_to(dx_feet_alip * cg_alip + dy_feet_alip * sg_alip == 0 );

    // 5. (可选但推荐) 最小站立宽度约束
    // 防止求解器为了偷懒，把两只脚重叠在一起放在终点 (虽然物理上不可能，但数学上是个可行解)
    double min_stance_width_alip = 0.15; // 最小允许两脚间距 15cm
    casadi::MX dist_sq_feet_alip = dx_feet_alip * dx_feet_alip + dy_feet_alip * dy_feet_alip;
    opti_alip.subject_to( dist_sq_feet_alip >= min_stance_width_alip * min_stance_width_alip );
    
    // 最大宽度约束通常由双圆运动学约束涵盖了，但为了保险也可以加一个
    double max_stance_width_alip = 0.30; 
    opti_alip.subject_to( dist_sq_feet_alip <= max_stance_width_alip * max_stance_width_alip );
#endif

    // casadi::MX P_final_com = P_alip(casadi::Slice(), N - 1);

    // 定义容差半径 (例如 1cm)
    // double final_pos_tolerance = 0.01; 

    // // 计算与目标的偏差
    // casadi::MX error_x = X_goal(0) + P_final_com(0) - x_com_goal(0);
    // casadi::MX error_y = X_goal(1) + P_final_com(1) - x_com_goal(1);

    // // 添加约束：偏差距离平方 <= 容差平方
    // opti_alip.subject_to( error_x * error_x + error_y * error_y <= final_pos_tolerance * final_pos_tolerance );

    // 4. 相邻步转角约束
    double max_step_yaw_alip = M_PI / 12; // 稍微放宽
    for (size_t i = 0; i < N - 1; ++i)
    {
        casadi::MX P_current = P_alip(casadi::Slice(), i);
        casadi::MX P_next = P_alip(casadi::Slice(), i + 1);
        opti_alip.subject_to(casadi::MX::abs(P_next(2) - P_current(2)) <= max_step_yaw_alip);
    }

     // 5. 运动学约束 (双圆约束)
    for (size_t i = 1; i < N; ++i) // 此编号为摆动脚的编号
    {
        bool cuurent_left_support;
        // 如果摆动脚是偶数步，那么支撑脚是奇数步，也就是支撑脚是右脚
        if (i % 2 == 0) cuurent_left_support = !initial_left_support; 
        else cuurent_left_support = initial_left_support;

        casadi::MX support_center = casadi::MX::vertcat({P_alip(0, i - 1), P_alip(1, i - 1), 0});
        casadi::MX yaw_support = P_alip(2, i - 1);
        
        double deta1 = 1.8; // 内圆圆心偏移量
        double deta2 = 0.35; // 外圆圆心偏移量
        double dis_th1 = 1.68; // 内圆半径
        double dis_th2 = 0.675; // 外圆半径

        casadi::MX P_next_pos = P_alip(casadi::Slice(), i);
        casadi::MX sin_yaw = casadi::MX::sin(yaw_support);
        casadi::MX cos_yaw = casadi::MX::cos(yaw_support);

        // 逻辑保持您的代码不变，仅修正变量类型
        if (!cuurent_left_support) { // Right leg support

            // inner circle
            casadi::MX p1_x = support_center(0) - (deta1) * sin_yaw;
            casadi::MX p1_y = support_center(1) + (deta1) * cos_yaw;
            casadi::MX dist1_sq = casadi::MX::pow(P_next_pos(0) - p1_x, 2) + casadi::MX::pow(P_next_pos(1) - p1_y, 2);
            opti_alip.subject_to(dist1_sq <= dis_th1 * dis_th1);
            
            // outer circle
            casadi::MX p2_x = support_center(0) - (-deta2) * sin_yaw;
            casadi::MX p2_y = support_center(1) + (-deta2) * cos_yaw;
            casadi::MX dist2_sq = casadi::MX::pow(P_next_pos(0) - p2_x, 2) + casadi::MX::pow(P_next_pos(1) - p2_y, 2);
            opti_alip.subject_to(dist2_sq <= dis_th2 * dis_th2);
            
        } else { // Left leg support
            // inner circle
            casadi::MX p1_x = support_center(0) - (-deta1) * sin_yaw;
            casadi::MX p1_y = support_center(1) + (-deta1) * cos_yaw;
            casadi::MX dist1_sq = casadi::MX::pow(P_next_pos(0) - p1_x, 2) + casadi::MX::pow(P_next_pos(1) - p1_y, 2);
            opti_alip.subject_to(dist1_sq <= dis_th1 * dis_th1);
            
            // outer circle
            casadi::MX p2_x = support_center(0) - (deta2) * sin_yaw;
            casadi::MX p2_y = support_center(1) + (deta2) * cos_yaw;
            casadi::MX dist2_sq = casadi::MX::pow(P_next_pos(0) - p2_x, 2) + casadi::MX::pow(P_next_pos(1) - p2_y, 2);
            opti_alip.subject_to(dist2_sq <= dis_th2 * dis_th2);
        }
    }   

    

    // // 猜测力矩为 0 (被动行走)
    opti_alip.set_initial(U, 0.0);

    opti_alip.solver("ipopt");

    // 使用上层的落脚点结果作为 ALIP 层的落脚点初始猜测
    for (int i = 0; i < N; ++i) {
        opti_alip.set_initial(P_alip(0, i), res_px[i]);
        opti_alip.set_initial(P_alip(1, i), res_py[i]);
        opti_alip.set_initial(P_alip(2, i), res_yaw[i]); 
    }   

    try {
        casadi::OptiSol sol = opti_alip.solve();
        std::cout << "ALIP Optimization Success!" << std::endl;
        
        // 简单打印结果验证
        res_px = std::vector<double>(sol.value(P_alip(0, casadi::Slice())));
        res_py = std::vector<double>(sol.value(P_alip(1, casadi::Slice())));
        res_yaw = std::vector<double>(sol.value(P_alip(2, casadi::Slice())));

        for(size_t i=0; i<res_px.size(); ++i) {
            std::cout << "Step " << i << ": (" << res_px[i] << ", " << res_py[i] << ", " <<res_yaw[i] << ")" << std::endl;
            Eigen::Vector2d sum_u = Eigen::Vector2d::Zero();
            double u_max = 0.0, v_max = 0.0;
            for (size_t j = 0; j < k; j++)
            {
                // std::cout << "  Control " << i << ": (" << sol.value(U(0, j + k * i)) << ", " << sol.value(U(1, j + k * i)) << ")" << std::endl;
                Eigen::Vector2d current_u(sol.value(U(0, j + k * i)), sol.value(U(1, j + k * i)));
                u_max = std::max(u_max, std::abs(current_u(0)));
                v_max = std::max(v_max, std::abs(current_u(1)));
                sum_u += current_u;
            }
            std::cout << std::fixed << std::setprecision(3);
            std::cout<< "  Average Control U: (" << sum_u(0)/k << ", " << sum_u(1)/k << ")" << std::endl;
            std::cout<< "  Max Control U: (" << u_max << ", " << v_max << ")" << std::endl;
        }
        auto end_time_alip = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_alip - start_time_alip).count();
        std::cout << "ALIP Optimization Time: " << duration << " ms" << std::endl;
        time_cost += duration;

    } catch(std::exception& e) {
        std::cerr << "ALIP Optimization Failed: " << e.what() << std::endl;
        // opti.debug().value(P);
    }
    
#endif
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
    
    // 画理想落脚点 (淡红色和淡蓝色箭头)
    std::vector<double> left_ideal_x, left_ideal_y, left_ideal_u, left_ideal_v;
    std::vector<double> right_ideal_x, right_ideal_y, right_ideal_u, right_ideal_v;

    // LOG(INFO)<<"left_foot_steps_ideal size: " << left_foot_steps_ideal.size();
    // LOG(INFO)<<"right_foot_steps_ideal size: " << right_foot_steps_ideal.size();
    // for(const auto& pt : left_foot_steps_ideal) {
    //     left_ideal_x.push_back(pt(0));
    //     left_ideal_y.push_back(pt(1));
    //     left_ideal_u.push_back(arrow_len * cos(pt(2)));
    //     left_ideal_v.push_back(arrow_len * sin(pt(2)));
    // }

    // for(const auto& pt : right_foot_steps_ideal) {
    //     right_ideal_x.push_back(pt(0));
    //     right_ideal_y.push_back(pt(1));
    //     right_ideal_u.push_back(arrow_len * cos(pt(2)));
    //     right_ideal_v.push_back(arrow_len * sin(pt(2)));
    // }
    // LOG(INFO)<<"drawn ideal footstep arrows.";

    // 画理想落脚点 (淡红色和淡蓝色点和箭头)
    // plt::scatter(left_ideal_x, left_ideal_y, 50.0, {{"color", "red"}, {"label", "Left Ideal"}});
    // plt::quiver(left_ideal_x, left_ideal_y, left_ideal_u, left_ideal_v, 
    //             {{"color", "lightcoral"}, {"alpha", "0.5"}});

    // plt::scatter(right_ideal_x, right_ideal_y, 50.0, {{"color", "blue"}, {"label", "Right Ideal"}});
    // plt::quiver(right_ideal_x, right_ideal_y, right_ideal_u, right_ideal_v, 
    //             {{"color", "lightblue"}, {"alpha", "0.5"}});
    // 淡红色箭头 (左脚理想位置)
    // plt::quiver(left_ideal_x, left_ideal_y, left_ideal_u, left_ideal_v, 
    //             {{"color", "lightcoral"}, {"alpha", "0.5"}, {"label", "Left Ideal"}});

    // 淡蓝色箭头 (右脚理想位置)
    // plt::quiver(right_ideal_x, right_ideal_y, right_ideal_u, right_ideal_v, 
                // {{"color", "lightblue"}, {"alpha", "0.5"}, {"label", "Right Ideal"}});
    LOG(INFO)<<"finish drawn ideal footstep arrows.";
    // 标记起点和终点
    // std::vector<double> start_x = {p_start_support_foot(0)};
    // std::vector<double> start_y = {p_start_support_foot(1)};
    // plt::text(start_x[0], start_y[0], "Start");
    
    std::vector<double> goal_pt_x = {x_goal};
    std::vector<double> goal_pt_y = {y_goal};
    plt::scatter(goal_pt_x, goal_pt_y, 100.0, {{"color", "green"}, {"marker", "*"}, {"label", "Goal"}});

    // 设置图形属性
    plt::title("Footstep Planning Result with ALIP " + std::to_string(N) + " Steps" +" duration: " + std::to_string(time_cost) + " ms");
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