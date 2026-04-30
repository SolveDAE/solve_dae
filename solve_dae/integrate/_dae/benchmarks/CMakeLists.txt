cmake_minimum_required(VERSION 3.10)
project(IDAExamples)

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

# - IDE
add_executable(arevalo weissinger/weissinger.c)
target_link_libraries(weissinger PRIVATE ${SUNDIALS_LIB})

