/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2025-12-04 14:33:53
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2025-12-05 13:45:10
 * @FilePath: /footstep_planner/src/QP_eigen_solver.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "osqp.h"
#include <iostream>
#include <Eigen/Sparse>
using namespace std;
#define degree2rad(x) x * M_PI/180.0;

struct Footstep
{
    bool is_left;
    double x, y, yaw;
    // 此处的变量个数应该与k相同
    vector<Eigen::Vector2d> u{10};
};

struct ConstraintBuilder {
    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<double> lower;
    std::vector<double> upper;
    int row_idx = 0; // 自动滚动的行号

    // 添加等式约束： coeffs * X = val
    // col_indices: 告诉函数这一行的系数对应大向量里的哪几列
    void add_equality(const std::vector<double>& coeffs, 
                      const std::vector<int>& col_indices, 
                      double val) {
        for(size_t i=0; i<coeffs.size(); ++i) {
            triplets.push_back(Eigen::Triplet<double>(row_idx, col_indices[i], coeffs[i]));
        }
        lower.push_back(val);
        upper.push_back(val);
        row_idx++; // 写完一行，行号+1
    }
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


int main(int argc, char** argv)
{
    // 输入约束
    Footstep footstep_initial; // 跟u没有关系
    footstep_initial.is_left = true;
    footstep_initial.x = 0;
    footstep_initial.y = 0.1;
    footstep_initial.yaw = 0.0;

    // 声明变量
    // 总计N*（3+k）个变量，k表示将摆动周期分解成多少段

    int N = 10, k = 10;
    double goal_yaw = degree2rad(45.0); // 目标朝向45度
    // 使用变量计算矩阵的实际维度 x、y、yaw、u1、u2、……、u10、v1、v2、……、v10
    // u为5关节y方向的力矩pitch，v为6关节x方向的力矩roll
    // int matrixDim = N * (3 + 2 * k);

    // 如果把所有的约束内和代价函数内所有用到的变量都参与优化，最后再提取出所需要的变量，那矩阵的维度为多少？
    // x y yaw (4+2)*(k) + 4(reset的状态量) 每一步的状态量 ，一共N步，再补上最后一个RESET状态。注意检查
    // 
    int fullMatrixDim = (3 + 6 * k + 4) * N + 4;

    // 声明一个稀疏矩阵，维度由变量计算得出
    Eigen::SparseMatrix<double> diagonalMatrix_u(fullMatrixDim, fullMatrixDim);
    diagonalMatrix_u.setZero();
    // 此处，将两个方向的力矩权重设置为相同
    double Q_u = 50; // 力矩的权重
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(fullMatrixDim); // 预留空间，提高效率
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 3; j < 3 + 6 * k; j++)
        {
            size_t dim = i * (3 + 6 * k + 4) + j;
            tripletList.push_back(Eigen::Triplet<double>(dim, dim, static_cast<double>(Q_u)));
        }
    }
    diagonalMatrix_u.setFromTriplets(tripletList.begin(), tripletList.end());

    // yaw的平滑性权重矩阵，保证两个相邻步的转角差的平方和最小
    Eigen::SparseMatrix<double> Matrix_yaw(fullMatrixDim, fullMatrixDim);
    Matrix_yaw.setZero();
    // std::vector<Eigen::Triplet<double>> tripletList_yaw;

    for (int i = 0; i < N - 1; i++)
    {
        int theta_i = i * (3 + 6 * k + 4) + 2;
        int theta_i_plus_1 = (i + 1) * (3 + 6 * k + 4) + 2;
        Matrix_yaw.coeffRef(theta_i, theta_i) += 1;
        Matrix_yaw.coeffRef(theta_i_plus_1, theta_i_plus_1) += 1;
        
        int cross_term_index1 = theta_i;
        int cross_term_index2 = theta_i_plus_1;
        Matrix_yaw.coeffRef(cross_term_index1, cross_term_index2) += -1;
        Matrix_yaw.coeffRef(cross_term_index2, cross_term_index1) += -1;
    }
    
    
    // 最终的H矩阵
    Eigen::SparseMatrix<double> H(fullMatrixDim, fullMatrixDim);
    H = 2 * (diagonalMatrix_u + Matrix_yaw); // 注意这里的系数2

    // 加上 终点状态接近程度这一项
    Eigen::SparseMatrix<double> Matrix_goal(fullMatrixDim, fullMatrixDim);
    Matrix_goal.setZero(); 
    double lambda_goal_yaw = 300;
    Matrix_goal.coeffRef((N - 1) * (3 + 6 * k + 4) + 0, (N - 1) * (3 + 6 * k + 4) + 0) += lambda_goal_yaw;
    Matrix_goal.coeffRef((N - 1) * (3 + 6 * k + 4) + 1, (N - 1) * (3 + 6 * k + 4) + 1) += lambda_goal_yaw;
    Matrix_goal.coeffRef((N - 1) * (3 + 6 * k + 4) + 2, (N - 1) * (3 + 6 * k + 4) + 2) += lambda_goal_yaw;
    Matrix_goal.coeffRef((N - 1) * (3 + 6 * k + 4) + 3, (N - 1) * (3 + 6 * k + 4) + 3) += lambda_goal_yaw;

    // yaw 脚朝向正确
    Matrix_goal.coeffRef((N - 2) * (3 + 6 * k + 4) + 2, (N - 2) * (3 + 6 * k + 4) + 2) += lambda_goal_yaw;

    H += 2 * Matrix_goal;

    double x_goal = 4.0;
    double y_goal = 0.0;
    // 终点离目标点的距离权重 这不就引入了常数项了吗？
    Eigen::VectorXd f = Eigen::VectorXd::Zero(fullMatrixDim);
    f.coeffRef((N - 1) * (3 + 6 * k + 4) + 0) = -2 * lambda_goal_yaw * x_goal; 
    f.coeffRef((N - 1) * (3 + 6 * k + 4) + 1) = -2 * lambda_goal_yaw * y_goal; 
    f.coeffRef((N - 2) * (3 + 6 * k + 4) + 2) = -2 * lambda_goal_yaw * goal_yaw; 
    

    // 上面已经将H矩阵和f向量计算完成，下面进行约束的构建，类似于模型预测控制
    ConstraintBuilder builder;
    // 两层循环，上层各之间，也就是reset内的循环。下层，单脚支撑期内的循环
    for (size_t i = 0; i < N; i++) // 外层循环，N个reset步
    {
        for (size_t j = 0; j < k; j++) // 内层循环，单脚支撑期内的k个子步
        {
            
        }

        // 外层循环的reset约束;注意还有最后一步到终点的状态约束
    }
    


    int nx = 0;
    int nu = 4;

    // for (int k = 0; k < N; ++k) {
    //     // 计算变量在全局向量中的起始索引
    //     int idx_x_k   = k * (nx + nu);          // x_k 的位置
    //     int idx_u_k   = idx_x_k + nx;           // u_k 的位置
    //     int idx_x_kp1 = (k + 1) * (nx + nu);    // x_{k+1} 的位置
    //     // 递推关系是向量方程，包含 nx 个标量等式，所以我们要添加 nx 行约束
    //     // 等式： x_{k+1}^{(i)} - (A*x_k)^{(i)} - (B*u_k)^{(i)} = 0
    //     for (int i = 0; i < nx; ++i) {
    //         // 每一个状态分量 i 都是一行约束
    //         std::vector<double> coeffs;
    //         std::vector<int> cols;
    //         // 1. 放入 -A 的第 i 行 (对应 x_k)
    //         for (int j = 0; j < nx; ++j) {
    //             coeffs.push_back(-A(i, j)); 
    //             cols.push_back(idx_x_k + j);
    //         }
    //         // 2. 放入 -B 的第 i 行 (对应 u_k)
    //         for (int j = 0; j < nu; ++j) {
    //             coeffs.push_back(-B(i)); // 这里B是向量，如果是矩阵就是 B(i,j)
    //             cols.push_back(idx_u_k + j);
    //         }
    //         // 3. 放入 +I (对应 x_{k+1})
    //         // x_{k+1} 的第 i 个元素系数是 1
    //         coeffs.push_back(1.0);
    //         cols.push_back(idx_x_kp1 + i);
    //         // 4. 提交这一行
    //         builder.add_equality(coeffs, cols, 0.0);
    //     }
    // }
    
    return 0;
}