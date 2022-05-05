add_executable(reduce_v0_256 reduce_v0_256.cu)
target_link_libraries(reduce_v0_256 ${EXTRALIBS})

add_executable(reduce_v0_512 reduce_v0_512.cu)
target_link_libraries(reduce_v0_512 ${EXTRALIBS})

add_executable(reduce_v0_1024 reduce_v0_1024.cu)
target_link_libraries(reduce_v0_1024 ${EXTRALIBS})

add_executable(reduce_v1_256 reduce_v1_256.cu)
target_link_libraries(reduce_v1_256 ${EXTRALIBS})

add_executable(reduce_v1_512 reduce_v1_512.cu)
target_link_libraries(reduce_v1_512 ${EXTRALIBS})

add_executable(reduce_v1_1024 reduce_v1_1024.cu)
target_link_libraries(reduce_v1_1024 ${EXTRALIBS})

add_executable(reduce_v2_256 reduce_v2_256.cu)
target_link_libraries(reduce_v2_256 ${EXTRALIBS})

add_executable(reduce_v2_512 reduce_v2_512.cu)
target_link_libraries(reduce_v2_512 ${EXTRALIBS})

add_executable(reduce_v2_1024 reduce_v2_1024.cu)
target_link_libraries(reduce_v2_1024 ${EXTRALIBS})