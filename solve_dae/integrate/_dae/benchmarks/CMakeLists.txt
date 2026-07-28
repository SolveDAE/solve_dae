cmake_minimum_required(VERSION 3.10)
project(IDA_Radau5_Examples)
enable_language(Fortran)

# Find the SUNDIALS package
find_package(SUNDIALS REQUIRED)

# Set sundials library
set(SUNDIALS_LIB SUNDIALS::core SUNDIALS::ida SUNDIALS::nvecserial -lm)

# Add executables

# - index 1 DAE
add_executable(brenan brenan/brenan.c)
target_link_libraries(brenan PRIVATE ${SUNDIALS_LIB})

# - index 1 DAE
add_executable(robertson robertson/robertson.c)
target_link_libraries(robertson PRIVATE ${SUNDIALS_LIB})

# - index 2 DAE
add_executable(knife_edge knife_edge/knife_edge.c)
target_link_libraries(knife_edge PRIVATE ${SUNDIALS_LIB})

# - index 3 DAE
add_executable(arevalo arevalo/arevalo.c)
target_link_libraries(arevalo PRIVATE ${SUNDIALS_LIB})

# - index 3 DAE
add_executable(particle particle/particle.c)
target_link_libraries(particle PRIVATE ${SUNDIALS_LIB})

# - IDE
add_executable(weissinger weissinger/weissinger.c)
target_link_libraries(weissinger PRIVATE ${SUNDIALS_LIB})

# ---------------------------------------------------------------------
# Fortran examples using E. Hairer's RADAU5
# ---------------------------------------------------------------------
# Shared RADAU5 solver sources (radau5.f, decsol.f, dc_decsol.f), reused by
# every *_radau5.f90 driver below -- add new drivers as separate
# executables linked against this one library, rather than copying the
# solver sources into each example directory.
add_library(radau5 STATIC
    radau5/radau5.f
    radau5/decsol.f
    radau5/dc_decsol.f
)

# - index 3 DAE (GGL / stabilized-index-2 reformulation, see particle/particle.f90)
add_executable(particle_radau5 particle/particle.f90)
target_link_libraries(particle_radau5 PRIVATE radau5)

