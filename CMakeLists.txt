cmake_minimum_required (VERSION 3.18.0)
project(cuda_optimizer  LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/bin")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)


cmake_policy(SET CMP0104 OLD)
set(CMAKE_CUDA_FLAGS "--compiler-options -fPIC -Wno-deprecated-gpu-targets")
foreach(arch 50 52 53 60 61 62 70 72 75 80)
    set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${arch},code=sm_${arch}")
endforeach()

option(ENABLE_OPENMP "enble openmp" OFF)


if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
    message("OpenMP FOUND: ${OpenMP_FOUND}")
endif()

find_package(Threads REQUIRED "SHARED")
find_package(CUDA REQUIRED "SHARED")
include_directories(${CUDA_INCLUDE_DIRS}) 


link_directories(${CUDA_DIR}/lib64)
link_directories(/usr/local/cuda/lib64/stubs/)
 
set(EXTRALIBS ${EXTRALIBS} glog gflags boost_system boost_filesystem rt pthread )
set(EXTRALIBS ${EXTRALIBS} cuda cudnn cublas cudart nvToolsExt ${CMAKE_DL_LIBS})


add_subdirectory(src)

