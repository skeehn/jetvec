cmake_minimum_required(VERSION 3.16)
project(jetvec C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(JETVEC_BUILD_CLI "Build CLI" ON)
option(JETVEC_BUILD_TESTS "Build tests" ON)

# Output dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Detect compiler
include(CheckCCompilerFlag)

# Sources
set(JETVEC_SRC
  src/api.c
  src/distance.c
  src/distance_scalar.c
  src/mmio.c
  src/util_crc64.c
)

# Try to add AVX2 object if supported (GCC/Clang)
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  check_c_compiler_flag("-mavx2" HAS_MAVX2)
  if(HAS_MAVX2)
    add_library(distance_avx2 OBJECT src/distance_avx2.c)
    target_compile_options(distance_avx2 PRIVATE -mavx2)
    list(APPEND JETVEC_SRC $<TARGET_OBJECTS:distance_avx2>)
    target_compile_definitions(distance_avx2 PRIVATE JETVEC_COMPILED_WITH_AVX2=1)
  endif()
endif()

add_library(jetvec STATIC ${JETVEC_SRC})
target_include_directories(jetvec PUBLIC include)
if(MSVC)
  target_compile_options(jetvec PRIVATE /O2)
else()
  target_compile_options(jetvec PRIVATE -O3 -fno-math-errno -ffast-math)
endif()

if(JETVEC_BUILD_CLI)
  add_executable(jv_cli src/cli.c)
  target_link_libraries(jv_cli PRIVATE jetvec)
endif()

if(JETVEC_BUILD_TESTS)
  enable_testing()
  add_executable(test_distance tests/test_distance.c)
  target_link_libraries(test_distance PRIVATE jetvec)
  add_test(NAME distance COMMAND test_distance)
endif()
