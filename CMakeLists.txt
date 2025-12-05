cmake_minimum_required(VERSION 3.8)
project(footstep_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED) # 查找 ROS 2 的 CMake 工具
find_package(matplotlib_cpp REQUIRED) # 查找 matplotlib_cpp 库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(OsqpEigen REQUIRED)

find_package(casadi REQUIRED)

find_package(glog REQUIRED)

find_package(Python3 3.10 EXACT COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 3.10 EXACT COMPONENTS NumPy)
find_package(PythonLibs REQUIRED)
include_directories( # 指定头文件搜索路径
    include
    ${Eigen3_inlude_directory} # 这里有拼写错误
    ${PYTHON_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${matplotlib_cpp_inlude_directory} # 这里有拼写错误
)

add_executable(footstep_planner # 添加一个可执行文件
    src/footstep_planner_casadi.cpp
)

target_link_libraries(footstep_planner # 链接库
    Eigen3::Eigen 
    ${PYTHON_LIBRARIES}
    ${matplotlib_cpp_library} # 这里有拼写错误
    casadi
    glog::glog
    # OsqpEigen::OsqpEigen
)

message(STATUS "Eigen3 found at: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "CasADi found version: ${casadi_VERSION}")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
