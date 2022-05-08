add_executable(copy_v0_32x8 copy_v0_32x8.cu)
target_link_libraries(copy_v0_32x8 ${EXTRALIBS})
add_executable(copy_v0_32x16 copy_v0_32x16.cu)
target_link_libraries(copy_v0_32x16 ${EXTRALIBS})
add_executable(copy_v0_32x32 copy_v0_32x32.cu)
target_link_libraries(copy_v0_32x32 ${EXTRALIBS})

