cmake_minimum_required(VERSION 3.16)
project(ingenuityml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
endif()

# Find Google Test (optional)
find_package(GTest QUIET)

# Include directories
include_directories(include)

# Collect all source files
file(GLOB_RECURSE INGENUITYML_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

# Create the core library
add_library(ingenuityml_core STATIC ${INGENUITYML_SOURCES})

# Find and configure OpenMP
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ingenuityml_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(ingenuityml_core PUBLIC ${OpenMP_CXX_FLAGS})
    add_compile_definitions(_OPENMP)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    set(OPENMP_AVAILABLE TRUE)
else()
    # Try to find OpenMP manually on macOS
    if(APPLE)
        target_compile_options(ingenuityml_core PUBLIC -Xpreprocessor -fopenmp)
        target_include_directories(ingenuityml_core PUBLIC /opt/homebrew/opt/libomp/include)
        target_link_libraries(ingenuityml_core PUBLIC -L/opt/homebrew/opt/libomp/lib -lomp)
        message(STATUS "OpenMP configured manually for macOS")
        set(OPENMP_AVAILABLE TRUE)
    else()
        message(WARNING "OpenMP not found - parallel processing will be disabled")
        set(OPENMP_AVAILABLE FALSE)
    endif()
endif()

target_link_libraries(ingenuityml_core PUBLIC Eigen3::Eigen)

# Create Python extension
pybind11_add_module(ingenuityml python/ingenuityml_bindings.cpp)
target_link_libraries(ingenuityml PRIVATE ingenuityml_core Eigen3::Eigen)

# Add OpenMP to Python extension if available
if(OPENMP_AVAILABLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ingenuityml PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(ingenuityml PRIVATE -L/opt/homebrew/opt/libomp/lib -lomp)
    endif()
endif()

# Optional: Add tests if GTest is available
if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()
