find_package(CUDA REQUIRED)

cuda_add_executable(device_query device_query.cpp)