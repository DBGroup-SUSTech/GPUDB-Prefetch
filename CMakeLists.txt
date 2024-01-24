cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES 86) # NVIDIA A10
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)


project(PrefetchSurvey LANGUAGES CXX CUDA)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# ---[ Using cmake scripts and modules
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Enable ASAN only as you need. Please notice
#       1. This project do not pass leak-santizer
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

find_package(GTest)
include(GoogleTest)
find_package(fmt)

# # ---[ Dependencies
# include(cmake/Dependencies.cmake)

set(SRC_PATH ${PROJECT_SOURCE_DIR}/src)
set(INC_PATH ${PROJECT_SOURCE_DIR}/include)
set(TEST_PATH ${PROJECT_SOURCE_DIR}/test)
message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB_RECURSE SRC_FILES "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE TEST_FILES "test/*.cu" "test/*.cpp")

foreach(test_file IN LISTS TEST_FILES)
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file} ${SRC_FILES})
        target_include_directories(${test_name}
                PUBLIC ${INC_PATH}
                PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                PUBLIC ${FMT_INCLUDE_DIR}
        )  
        target_link_libraries(${test_name} 
                PRIVATE cuda 
                PRIVATE GTest::gtest_main
                PRIVATE fmt::fmt
        ) 
        target_compile_options(${test_name} PRIVATE --ptxas-options=-v)

        gtest_discover_tests(${test_name})
endforeach()
