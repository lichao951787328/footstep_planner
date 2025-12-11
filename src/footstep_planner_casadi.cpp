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

    Eigen::Matrix<double, 4, 3> B_cop;
    B_cop <<0,0,0,
            0,0,0,
            0,mass * g,0,
            -mass * g,0,0;
    Eigen::Matrix<double, 4, 3> B_ds = Ar_ds * A_c.inverse() * (A_c.inverse() * (Eigen::Matrix4d::Identity() - Ar_ds.inverse())/T_ds - Ar_ds.inverse()) * B_cop;
    Eigen::Matrix<double, 4, 3> B_r = B_ds + P_map;
    return {Ar_ds, B_r};
}

struct ProfileParams {
    double v_start = 0.0; // 假设从静止开始规划参考轨迹
    double v_end = 0.0; // 假设从静止开始规划参考轨迹
    double v_max;         // 允许的最大巡航速度
    double a_max;         // 最大加速度
    double a_min;         // 最大减速度 (建议设为正值)
    double total_dist;    // 总距离
    double total_time;    // 规划器分配的总时间 (N-1)*dt
    double dt;            // 单步时间
};
// to do list: 存在的问题，如果v_start>v_max怎么办？暂时不考虑这种情况
std::vector<double> generate_motion_profile(ProfileParams p, int steps) {
    std::vector<double> ref_positions;
    
    // --- 第一步：物理极限计算 (Physics Check) ---
    
    // 计算加速到 v_max 所需的距离
    // d = (v_f^2 - v_i^2) / (2*a)
    double d_acc_full = (p.v_max * p.v_max - p.v_start * p.v_start) / (2.0 * p.a_max);
    // 计算从 v_max 减速到 0 所需的距离
    double d_dec_full = (p.v_max * p.v_max) / (2.0 * p.a_min);
    
    double v_peak, t_acc, t_dec, t_cruise;
    
    // --- 第二步：判断是 三角形 还是 梯形 ---
    if (p.total_dist < (d_acc_full + d_dec_full)) {
        // [情况 A: 三角形规划] 距离太短，达不到 v_max
        // 此时没有匀速段，加速完立刻减速
        // d_total = d_acc + d_dec = v_peak^2 / (2*a_acc) + v_peak^2 / (2*a_dec)
        // 解方程得 v_peak:
        v_peak = std::sqrt((2.0 * p.total_dist) / (1.0/p.a_max + 1.0/p.a_min));
        
        t_acc = v_peak / p.a_max;
        t_dec = v_peak / p.a_min;
        t_cruise = 0.0;
        
    } else {
        // [情况 B: 梯形规划] 距离足够，可以达到 v_max
        v_peak = p.v_max;
        
        t_acc = v_peak / p.a_max;
        t_dec = v_peak / p.a_min;
        
        double d_cruise = p.total_dist - (d_acc_full + d_dec_full);
        t_cruise = d_cruise / v_peak;
    }
    
    // 物理上走完这段路所需的“最快时间”
    double t_total_phys = t_acc + t_cruise + t_dec;

    LOG(INFO)<< "t_acc: "<<t_acc<<", t_cruise: "<<t_cruise<<", t_dec: "<<t_dec<<", t_total_phys: "<<t_total_phys;
    // --- 第三步：时间缩放 (Time Scaling) ---
    // 关键点：ALIP 规划器的步数 N 和单步时间 T 是固定的，总时间 p.total_time 是硬约束。
    // 物理计算出的 t_total_phys 可能比 p.total_time 短（说明 N 给多了），也可能长（说明 N 给少了）。
    // 我们需要把物理曲线“拉伸”或“压缩”来适配 p.total_time。
    
    // 首先明确规划的第一步是起始步，不计算在内，最后一步是在抵达终点后的并步，也不计算在内
    
    // 这里的steps 怎么和我们设定的重合？？
    // int steps = std::round(p.total_time / p.dt) + 1;

    // 这里应该是两头夹住不动，中间拉伸/压缩
    ref_positions.resize(steps); 
    ref_positions.at(0) = 0.0; // 起始位置
    ref_positions.at(steps - 2) = p.total_dist; // 终止位置
    ref_positions.at(steps - 1) = p.total_dist; // 终止位置

    // 先填充加速度段和减速段，根据步态周期和加速减速时间计算每一步对应的参考位置
    // 确定要几步实现加速
    int steps_acc = std::ceil(t_acc / p.dt);
    int steps_dec = std::ceil(t_dec / p.dt);
    for (int i = 1; i <= steps_acc; i++)
    {
        double t_planner = i * p.dt;
        // 映射到物理曲线上的时间点
        double t = t_planner * (t_total_phys / p.total_time);
        double s = 0.5 * p.a_max * t * t; // s = 0.5*a*t^2
        ref_positions.at(i) = s;
    }

    for (int i = 1; i <= steps_dec; i++)
    {
        // 从后往前填充，把减速过程看成是从终点往回走的加速过程
        double t_planner = i * p.dt;
        // 映射到物理曲线上的时间点
        double t = t_planner * (t_total_phys / p.total_time);
        // double t_d = t_dec - t; // 减速段的剩余时间
        double s_dec_from_end = 0.5 * p.a_min * t * t; // 从终点往回算的距离
        ref_positions.at(steps - 2 - i) = p.total_dist - s_dec_from_end;
    }
    
    // 填充中间的匀速段，对中间还剩的时间段进行均匀填充
    int start_idx = steps_acc + 1;
    int end_idx = steps - 3 - steps_dec;
    int cruise_steps = end_idx - start_idx + 1; // + 1 是怎么来的
    LOG(INFO)<<"start_idx: "<<start_idx<<", end_idx: "<<end_idx;
    LOG(INFO)<< "steps_acc: " << steps_acc << ", steps_dec: " << steps_dec << ", cruise_steps: " << cruise_steps;

    // 保证均匀分布的起点为加速段的终点，终点为减速段的起点
    double s_start = ref_positions.at(steps_acc); // 加速截止位置
    double s_end = ref_positions.at(steps - 2 - steps_dec);

    for (int i = 1; i <= cruise_steps; ++i) {
        double ratio = static_cast<double>(i) / (cruise_steps + 1);
        ref_positions.at(start_idx + i - 1) = s_start + ratio * (s_end - s_start);
    }
    // Alternative implementation:
    // double s_start = ref_positions.at(steps_acc);
    // double s_end = ref_positions.at(steps - 2 - steps_dec);
    // double delta_s = s_end - s_start;
    // double v_cruise = delta_s / (cruise_steps * p.dt);
    // LOG(INFO)<< "Cruise steps: " << cruise_steps << ", delta_s: " << delta_s << ", v_cruise: " << v_cruise;


    // for (int i = 0; i < cruise_steps; ++i) {
    //     double ratio = static_cast<double>(i) / (cruise_steps - 1);
    //     ref_positions.at(start_idx + i) = ref_positions.at(steps_acc) + ratio * (ref_positions.at(steps - 2 - steps_dec) - ref_positions.at(steps_acc));
    // }
    for (auto & pos : ref_positions)
    {
        LOG(INFO)<< "Ref position: " << pos;
    }
    
    // ref_positions.push_back(0.0); // Step 0
    // // 不包括起始步和终止步，中间步数为 steps - 2
    // for (int i = 1; i < steps - 1; ++i) {
    //     // 当前 ALIP 规划器的时间点
    //     double t_planner = i * p.dt;
    //     // 映射到物理曲线上的时间点
    //     // 比例关系：t_phys / t_total_phys = t_planner / p.total_time
    //     double t = t_planner * (t_total_phys / p.total_time);
    //     LOG(INFO)<< "At planner step " << i << ", t_planner: " << t_planner << ", mapped t_phys: " << t;
    //     double s = 0.0;
    //     if (t <= t_acc) {
    //         // 加速阶段: s = 0.5 * a * t^2
    //         s = 0.5 * p.a_max * t * t;
    //         LOG(INFO)<< "  -> in acceleration phase.";
    //     } 
    //     else if (t <= (t_acc + t_cruise)) {
    //         // 匀速阶段: s = s_acc + v * dt
    //         double s_acc = 0.5 * p.a_max * t_acc * t_acc;
    //         s = s_acc + v_peak * (t - t_acc);
    //         LOG(INFO)<< "  -> in cruising phase.";
    //     } 
    //     else if (t <= t_total_phys) {
    //         // 减速阶段
    //         double s_acc = 0.5 * p.a_max * t_acc * t_acc;
    //         double s_cruise = v_peak * t_cruise;
    //         double t_d = t - (t_acc + t_cruise); // 已经在减速段走了多久
    //         // s = s_before + v*t - 0.5*a*t^2
    //         s = s_acc + s_cruise + (v_peak * t_d - 0.5 * p.a_min * t_d * t_d);
    //         LOG(INFO)<< "  -> in deceleration phase.";
    //     } 
    //     else {
    //         // 已经到达终点
    //         LOG(INFO)<< "  -> reached the end.";
    //         s = p.total_dist;
    //     }
    //     // 钳位防止数值误差
    //     if (s > p.total_dist) s = p.total_dist;
    //     LOG(INFO)<< "  -> ref position s: " << s;
    //     ref_positions.push_back(s);
    // }
    
    return ref_positions;
}

// 定义一个简单的结构体存路径点
struct PathPoint {
    double s; // 距离起点的累计弧长
    double x; // 全局 x
    double y; // 全局 y
    double yaw; // 切线角度 (可选，用于朝向引导)
    double accum_angle; // 累计角度 (可选)
};

// 离散化多项式曲线，生成查找表
std::vector<PathPoint> discretize_polynomial_curve(
    const Eigen::Vector<double, 6>& param, 
    double x_start, 
    double x_end, 
    double step_size = 0.01) // 1cm 的精度足够了
{
    std::vector<PathPoint> path;
    
    double current_x = x_start;
    double accum_dist = 0.0;
    double accum_angle = 0.0;
    // 放入起点
    double current_y = polynomial_func(param, current_x);
    // 计算起点的切线角 (简单差分法，或者求导)
    double next_y_temp = polynomial_func(param, current_x + 0.001);
    double start_yaw = atan2(next_y_temp - current_y, 0.001);
    
    path.push_back({0.0, current_x, current_y, start_yaw, 0.0});
    
    // 循环积分
    while (current_x < x_end) {
        double next_x = current_x + step_size;
        if (next_x > x_end) next_x = x_end; // 防止超调
        
        double next_y = polynomial_func(param, next_x);
        
        // 计算这一小段的直线距离
        double dx = next_x - current_x;
        double dy = next_y - current_y;
        double ds = std::sqrt(dx*dx + dy*dy);
        
        accum_dist += ds;
        double yaw = atan2(dy, dx);
        double yaw_diff = yaw - path.back().yaw;
        while (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
        while (yaw_diff < -M_PI) yaw_diff += 2 * M_PI;
        accum_angle += abs(yaw_diff);
        path.push_back({accum_dist, next_x, next_y, yaw, accum_angle});
        
        current_x = next_x;
        current_y = next_y;
    }
    return path;
}

// 查表插值函数：给定距离 s，返回 (x, y, yaw)
Eigen::Vector3d get_pose_from_s(const std::vector<PathPoint>& path, double s_req) {
    // 1. 处理边界情况
    if (path.empty()) return Eigen::Vector3d(0, 0, 0);
    if (s_req <= 0) return Eigen::Vector3d(path.front().x, path.front().y, path.front().yaw);
    if (s_req >= path.back().s) return Eigen::Vector3d(path.back().x, path.back().y, path.back().yaw);
    
    // 2. 二分查找找到 s_req 所在的区间
    // lower_bound 返回第一个 s >= s_req 的迭代器
    auto it = std::lower_bound(path.begin(), path.end(), s_req, 
        [](const PathPoint& p, double val) {
            return p.s < val;
        });
    
    // it 指向的是后一个点 (next)，it-1 是前一个点 (prev)
    const PathPoint& next = *it;
    const PathPoint& prev = *(it - 1);
    
    // 3. 线性插值
    double ratio = (s_req - prev.s) / (next.s - prev.s);
    double x = prev.x + ratio * (next.x - prev.x);
    double y = prev.y + ratio * (next.y - prev.y);
    double yaw = prev.yaw + ratio * (next.yaw - prev.yaw);
    
    return Eigen::Vector3d(x, y, yaw);
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

    ProfileParams props;
    props.v_start = 0.0;
    props.v_end = 0.0;
    props.v_max = 0.3; // 设定一个合理的巡航速度 (m/s)
    props.a_max = 0.2; // 设定最大加速度
    props.a_min = 0.2; 

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

    std::vector<PathPoint> path_lut = discretize_polynomial_curve(
        polynomial_param, 
        p_start_support_foot(0), // start_x
        x_goal,                  // end_x
        0.3                    // 精度 1cm
    );

    // for (auto & path_point : path_lut)
    // {
    //     cout << "s: " << path_point.s << ", x: " << path_point.x << ", y: " << path_point.y << ", yaw: " << path_point.yaw << ", accum_angle: " << path_point.accum_angle << endl;  
    // }
    

    // 更新 total_distance 为曲线的实际长度，而不是直线距离
    // 这样速度规划会更准确
    // double curve_length = ;
    
    // [重新生成 s_ref] 因为总距离变了 (曲线比直线长)
    // props.total_dist = std::sqrt(std::pow(x_goal - p_start_support_foot(0), 2) + std::pow(y_goal - p_start_support_foot(1), 2));
    props.total_dist = path_lut.back().s; // 使用曲线的实际长度
    LOG(INFO)<< "Total path distance (straight line): " << props.total_dist;

    LOG(INFO) << "line distance: " << std::sqrt(std::pow(x_goal - p_start_support_foot(0), 2) + std::pow(y_goal - p_start_support_foot(1), 2));
    // props.total_time 保持不变 (由 N 决定)
    

    // size_t steps_1 = std::ceil((std::sqrt(std::pow(x_goal - p_start_support_foot(0), 2) + std::pow(y_goal - p_start_support_foot(1), 2))) / (0.3)) + 2; 
    size_t steps_1 = std::ceil(props.total_dist / (0.3)) + 2; 
    LOG(INFO)<< "Determined number of steps based on distance: " << steps_1;
    // double start_x = 0.0; // 修正: 假设从 0 开始
    // double total_yaw_change = 0.0;
    // double sample_step = 0.1;
    // double prev_yaw = 0.0; 
    
    // for (double x = start_x + sample_step; x <= x_goal; x += sample_step) {
    //     double y = polynomial_func(polynomial_param, x);
    //     double y_next = polynomial_func(polynomial_param, x + delta_x);
    //     double current_yaw = atan2(y_next - y, delta_x);
    //     double yaw_diff = current_yaw - prev_yaw;
    //     while (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
    //     while (yaw_diff < -M_PI) yaw_diff += 2 * M_PI; 
    //     total_yaw_change += std::abs(yaw_diff);
    //     prev_yaw = current_yaw;
    // }
    
    double total_yaw_change = path_lut.back().accum_angle;

    size_t steps_2 = std::ceil(total_yaw_change / (M_PI / 12)); 
    LOG(INFO)<< "Determined number of steps based on yaw change: " << steps_2;
    size_t N = std::max(steps_1, steps_2) ;
    LOG(INFO)<< "Determined number of steps N: " << N;

    if (N > 20)
    {
        LOG(WARNING)<< "Planned steps N is large (" << N << "), may lead to high computation time.";
    }


#ifdef ALIP
    // 动力学参数
    size_t k = 10; // 将单脚支撑期分成了k段，也就是从单脚支撑期开始到单脚脚支撑期结束共有k+1个时刻，其实每个步态也就是这k+1个时刻，因为双脚支撑的末尾正好对应下一次单脚支撑期开始状态。在这k+1个时刻中，只有k个时刻有控制输入。
    double H_com = 0.9, mass = 60, g = 9.81;
    double swing_t = 0.8, double_support = 0.2;
    double T_ss_dt = swing_t / k;
    
    double step_duration = swing_t + double_support; // 0.8 + 0.2
    props.dt = step_duration;
    props.total_time = (N - 2) * step_duration;
    LOG(INFO)<< "Updated total_time for profile generation: " << props.total_time;

    std::vector<double> s_ref = generate_motion_profile(props, N);

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
#endif

    // 3. 优化器设置
    casadi::Opti opti = casadi::Opti();

    // --- 变量定义 (顺序不变) ---
#ifdef ALIP
    casadi::MX X = opti.variable(4, N * (k + 1)); // 包含了初始状态，没有包含终点状态
    casadi::MX X_goal = opti.variable(4); // 终点状态变量
    // X_goal(2) = 0; X_goal(3) = 0; // 终点状态设为零
    casadi::MX U = opti.variable(2, N * k); // 控制输入变量
#endif
    casadi::MX P = opti.variable(3, N); 

    // --- 代价函数 (顺序不变) ---
    casadi::MX J = 0;
#ifdef ALIP 
    // 最小化输入
    double lambda_u = 2.0;
    for (size_t i = 0; i < N * k; ++i) {
        J += lambda_u * (U(0, i) * U(0, i) + U(1, i) * U(1, i));
    }

    // 终点约束 Cost
    double lambda_goal_yaw = 50000.0;
    
    // [修正]: ALIP 的 X 是局部状态。全局位置 = P_final + X_final_local
    // 我们用最后一步的 P 和 X_goal 来计算全局位置
    casadi::MX P_final_com = P(casadi::Slice(), N - 1);
    
    J += lambda_goal_yaw * ((X_goal(0) + P_final_com(0) - x_com_goal(0)) * (X_goal(0) + P_final_com(0) - x_com_goal(0)) + (X_goal(1) + P_final_com(1) - x_com_goal(1)) * (X_goal(1) + P_final_com(1) - x_com_goal(1)));
#endif

    // 转角均匀 Cost
    double lambda_yaw_smooth = 10.0;
    for (size_t i = 0; i < N - 1; ++i) {
        J += lambda_yaw_smooth * (P(2, i + 1) - P(2, i)) * (P(2, i + 1) - P(2, i));
    }
#ifndef ALIP 
    // 步态均匀性
    double lambda_step_smooth = 5.0; // 这个不作强制性约束，因为在引入动力学后这一项可由动力学保证
    for (size_t i = 0; i < N - 1; i++)
    {
        casadi::MX P_current = P(casadi::Slice(), i);
        casadi::MX P_next = P(casadi::Slice(), i + 1);
        J += lambda_step_smooth * ((P_next(0) - P_current(0)) * (P_next(0) - P_current(0)) + (P_next(1) - P_current(1)) * (P_next(1) - P_current(1)));
    }
#endif
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
#ifdef ALIP 
    // double lambda_progress = 40.0;
    // for (size_t i = 0; i < N; ++i) {
    //     // 简单的线性插值期望位置
    //     double expected_x = p_start_support_foot(0) + (double)i / (N - 1) * (x_goal - p_start_support_foot(0));
    //     J += lambda_progress * casadi::MX::pow(P(0, i) - expected_x, 2);
    // }

    double lambda_profile_tracking = 200.0; // 强力跟踪参考轨迹

    vector<Eigen::Vector3d> left_foot_steps_ideal;
    vector<Eigen::Vector3d> right_foot_steps_ideal;

    for (auto & ref_point : s_ref)
    {
        LOG(INFO)<< "Reference s: " << ref_point;
    }
    

    for (size_t i = 0; i < N; ++i)
    {
        // 1. 从速度规划中拿到这一步应该走的累计距离 s
        double current_s = s_ref[i];
        
        // 2. 从曲线表中查出这个 s 对应的 (x, y)
        Eigen::Vector3d ideal_pos = get_pose_from_s(path_lut, current_s);
        LOG(INFO)<< "Ideal position: " << ideal_pos.transpose();
        double ideal_x = ideal_pos(0);
        double ideal_y = ideal_pos(1);
        double ideal_yaw = ideal_pos(2);

        // 判断左右脚
        bool is_left_foot;
        if (initial_left_support)
        {
            // 左脚支撑，落脚点在右侧
            if (i%2 == 0) // 对应的是右侧落脚点
            {
                is_left_foot = true;
            }
            else // 对应的是左侧落脚点
            {
                is_left_foot = false;
            }
        }
        else
        {
            if (i%2 == 0) // 对应的是左侧落脚点
            {
                is_left_foot = false;
            }
            else // 对应的是右侧落脚点
            {
                is_left_foot = true;
            }
        }
        LOG(INFO)<< "Step " << i << " is " << (is_left_foot ? "left" : "right") << " foot.";
        double foot_offset_y = 0.1 * (is_left_foot ? 1 : -1); // 10cm 的脚间距
        ideal_x -= foot_offset_y * sin(ideal_yaw);
        ideal_y += foot_offset_y * cos(ideal_yaw);

        // debug 存储理想落脚点
        if (is_left_foot)
        {
            left_foot_steps_ideal.push_back(Eigen::Vector3d(ideal_x, ideal_y, ideal_yaw));
        }
        else
        {
            right_foot_steps_ideal.push_back(Eigen::Vector3d(ideal_x, ideal_y, ideal_yaw));
        }
        LOG(INFO)<< "Step " << i << " ideal foot position: (" << ideal_x << ", " << ideal_y << "), yaw: " << ideal_yaw; 
        // 3. 添加 Cost：让实际落脚点去追踪这个理想曲线上的点
        J += lambda_profile_tracking * (casadi::MX::pow(P(0, i) - ideal_x, 2) + 
                                        casadi::MX::pow(P(1, i) - ideal_y, 2));

        // 4. 添加 Cost：让实际落脚点的朝向去追踪理想切线角
        // J += lambda_profile_tracking * (1.0 - casadi::MX::cos(P(2, i) - ideal_yaw));
    }
#endif

#ifdef ALIP
    double lambda_terminal_state = 1000.0; // 权重很大，效果接近硬约束
    
    // 惩罚末端位置偏差 (防止摔倒)
    J += lambda_terminal_state * (casadi::MX::pow(X_goal(0), 2) + casadi::MX::pow(X_goal(1), 2));
    
    // 惩罚末端动量 (刹车)
    J += lambda_terminal_state * (casadi::MX::pow(X_goal(2), 2) + casadi::MX::pow(X_goal(3), 2));

#endif   

    opti.minimize(J);

    // --- 约束定义 (顺序不变) ---
    // 1. ALIP 动力学与 Reset Map (逻辑保持不变，仅修正矩阵乘法语法)
#ifdef ALIP 
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
            opti.subject_to(X_next == casadi::MX::mtimes(A_d_mpc_dm, X_current) + casadi::MX::mtimes(B_d_mpc_dm, U_current));
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
                P_current = casadi::MX::vertcat({P(0, i - 1), P(1, i - 1), 0}); // 假设 yaw 影响忽略或 P(2)
            }
            casadi::MX P_next_step = casadi::MX::vertcat({P(0, i), P(1, i), 0}); 
            // X_next = Ar * X_curr + Br * (P_new - P_old)
            // 注意：Reset 时，P_new 是刚才迈出去落地的脚 P(i)，P_old 是刚才的支撑脚
            opti.subject_to(X_next_state == casadi::MX::mtimes(Ar_reset_dm, X_current) + casadi::MX::mtimes(Br_reset_dm, P_next_step - P_current));
        }
    }
    
    // 2. 初始状态约束
    // [修正]: ALIP 模型中，X 是相对于支撑脚的局部状态。
    // 如果初始时刻机器人在原点 (0,0)，初始支撑脚在 (0, 0.1)，那么初始局部位置应该是 (0, -0.1)
    // 否则机器人第一步就会因为没有倾倒力矩而没法动
    // 这一点是根据初始支撑脚是左右脚来决定的
    /**
     * @brief 这个初始状态非常重要，因为 ALIP 模型的状态是相对于支撑脚的位置。尤其是角动量部分。
     * 
     */
    casadi::MX X_initial = X(casadi::Slice(), 0);
    double v_start_expected = 0.3; // 0.3 m/s 的启动速度
    double Ly_start = mass * H_com * v_start_expected;

    // opti.subject_to(X_initial(0) == 0.04); 
    // opti.subject_to(X_initial(1) == -0.03); // 保持 y 轴相对位置
    // opti.subject_to(X_initial(2) == 0); // Lx 依然可以是 0 (假设还没开始侧摆)
    // opti.subject_to(X_initial(3) == Ly_start); // <--- 给定初速度

    opti.subject_to(X_initial(0) == 0); 
    opti.subject_to(X_initial(1) == -0.01); // 保持 y 轴相对位置
    opti.subject_to(X_initial(2) == 0); // Lx 依然可以是 0 (假设还没开始侧摆)
    opti.subject_to(X_initial(3) == 0.5); // <--- 给定初速度
#endif
    // 初始落脚点约束
    casadi::MX P_initial = P(casadi::Slice(), 0);
    opti.subject_to(P_initial(0) == p_start_support_foot(0));
    opti.subject_to(P_initial(1) == p_start_support_foot(1));
    opti.subject_to(P_initial(2) == 0); // 假设初始朝向为0
#ifdef ALIP 
    // 保证机器人前两步往前走
    // opti.subject_to(P(casadi::Slice(), 1)(0) >= P(casadi::Slice(), 0)(0) + 0.02);
    // opti.subject_to(P(casadi::Slice(), 2)(0) >= P(casadi::Slice(), 1)(0) + 0.1);

    // 3. 步长几何约束 (分段策略)
    // for (size_t i = 0; i < N - 1; ++i)
    // {
    //     opti.subject_to(P(casadi::Slice(), i + 1)(0) >= P(casadi::Slice(), i)(0));
    // }

    // opti.subject_to(opti.bounded(-0.02, X_goal(0), 0.02));
    // opti.subject_to(opti.bounded(-0.02, X_goal(1), 0.02));
#endif 
    // 终点约束
    // 新的终点约束方案：中点重合 + 连线垂直
    // ==========================================
    // 1. 获取最后两步的状态变量
#ifdef GOAL_Hard_Constraint
    casadi::MX P_final = P(casadi::Slice(), N - 1); // 最后一步落脚点
    casadi::MX P_prev  = P(casadi::Slice(), N - 2); // 倒数第二步落脚点

    // 2. 约束两脚的朝向都必须对齐目标 Yaw
    //    (虽然是连线垂直，但脚本身的朝向还是应该朝前的)
    opti.subject_to(P_final(2) == yaw_goal);
    opti.subject_to(P_prev(2)  == yaw_goal);

    casadi::MX dx_feet = P_final(0) - P_prev(0);
    casadi::MX dy_feet = P_final(1) - P_prev(1);

// #ifndef ALIP
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
// #endif
    // 5. (可选但推荐) 最小站立宽度约束
    // 防止求解器为了偷懒，把两只脚重叠在一起放在终点 (虽然物理上不可能，但数学上是个可行解)
    double min_stance_width = 0.15; // 最小允许两脚间距 15cm
    casadi::MX dist_sq_feet = dx_feet * dx_feet + dy_feet * dy_feet;
    opti.subject_to( dist_sq_feet >= min_stance_width * min_stance_width );
    
    // 最大宽度约束通常由双圆运动学约束涵盖了，但为了保险也可以加一个
    double max_stance_width = 0.30; 
    opti.subject_to( dist_sq_feet <= max_stance_width * max_stance_width );
#endif
    // 3. 力矩约束
#ifdef ALIP 
    double max_roll = 50.0; // 稍微放宽一点，避免无解
    double max_pitch = 50.0;
    for (size_t i = 0; i < N * k; ++i)
    {
        casadi::MX U_current = U(casadi::Slice(), i);
        // 使用 norm_2
        double limit_sq = max_roll * max_roll + max_pitch * max_pitch;
        // 使用 sumsqr (u_x^2 + u_y^2) 替代 norm_2
        opti.subject_to( U_current(0)*U_current(0) + U_current(1)*U_current(1) <= limit_sq );
    }
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

#ifdef ALIP
    // // 猜测力矩为 0 (被动行走)
    opti.set_initial(U, 0.0);
#endif

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
#ifdef ALIP
        control_u = std::vector<Eigen::Vector2d>(sol.value(U).size2());
        for(size_t i=0; i<res_px.size(); ++i) {
            std::cout << std::fixed << std::setprecision(3);
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
#else
        for(size_t i=0; i<res_px.size(); ++i) {
            std::cout << "Step " << i << ": (" << res_px[i] << ", " << res_py[i] << ", " <<res_yaw[i] << ")" << std::endl;
        }
#endif
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Optimization Time: " << duration << " ms" << std::endl;
        time_cost = duration;

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
    
    // 画理想落脚点 (淡红色和淡蓝色箭头)
    std::vector<double> left_ideal_x, left_ideal_y, left_ideal_u, left_ideal_v;
    std::vector<double> right_ideal_x, right_ideal_y, right_ideal_u, right_ideal_v;

    LOG(INFO)<<"left_foot_steps_ideal size: " << left_foot_steps_ideal.size();
    LOG(INFO)<<"right_foot_steps_ideal size: " << right_foot_steps_ideal.size();
    for(const auto& pt : left_foot_steps_ideal) {
        left_ideal_x.push_back(pt(0));
        left_ideal_y.push_back(pt(1));
        left_ideal_u.push_back(arrow_len * cos(pt(2)));
        left_ideal_v.push_back(arrow_len * sin(pt(2)));
    }

    for(const auto& pt : right_foot_steps_ideal) {
        right_ideal_x.push_back(pt(0));
        right_ideal_y.push_back(pt(1));
        right_ideal_u.push_back(arrow_len * cos(pt(2)));
        right_ideal_v.push_back(arrow_len * sin(pt(2)));
    }
    LOG(INFO)<<"drawn ideal footstep arrows.";

    // 画理想落脚点 (淡红色和淡蓝色点和箭头)
    plt::scatter(left_ideal_x, left_ideal_y, 50.0, {{"color", "red"}, {"label", "Left Ideal"}});
    // plt::quiver(left_ideal_x, left_ideal_y, left_ideal_u, left_ideal_v, 
    //             {{"color", "lightcoral"}, {"alpha", "0.5"}});

    plt::scatter(right_ideal_x, right_ideal_y, 50.0, {{"color", "blue"}, {"label", "Right Ideal"}});
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