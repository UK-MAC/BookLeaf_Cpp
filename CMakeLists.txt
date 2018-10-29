# 3.10 required for modernised FindMPI
cmake_minimum_required(VERSION 3.10)

project(
    bookleaf
    VERSION 2.0.2
    DESCRIPTION "BookLeaf is a 2D unstructured hydrodynamics mini-app."
    LANGUAGES C CXX
)



# Check platform
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    add_definitions("-DBOOKLEAF_LINUX_BUILD")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions("-DBOOKLEAF_DARWIN_BUILD")
endif ()



# Build type defaults to Release
if (NOT DEFINED CMAKE_BUILD_TYPE)
    message(STATUS "Defaulting to Release build as build type was not specified")
    set(CMAKE_BUILD_TYPE "Release")
endif (NOT DEFINED CMAKE_BUILD_TYPE)

# Locate custom Find*.cmake files
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Used in custom Find*.cmake files
include(FindPackageHandleStandardArgs)

# Set search paths
string(REPLACE ":" ";" LIBRARY_DIRS "$ENV{LD_LIBRARY_PATH}")
list(APPEND CMAKE_LIBRARY_PATH ${LIBRARY_DIRS})

string(REPLACE ":" ";" INCLUDE_DIRS "$ENV{C_INCLUDE_PATH}")
list(APPEND CMAKE_INCLUDE_PATH ${INCLUDE_DIRS})

string(REPLACE ":" ";" INCLUDE_DIRS "$ENV{CPLUS_INCLUDE_PATH}")
list(APPEND CMAKE_INCLUDE_PATH ${INCLUDE_DIRS})



# Extra features off by default
# ... Typhon (MPI support)
if (NOT DEFINED ENABLE_TYPHON)
    set(ENABLE_TYPHON OFF)
endif (NOT DEFINED ENABLE_TYPHON)

# ... Silo (visualisation dumps)
if (NOT DEFINED ENABLE_SILO)
    set(ENABLE_SILO OFF)
endif (NOT DEFINED ENABLE_SILO)

# ... ParMETIS (distributed mesh partitioning)
if (NOT DEFINED ENABLE_PARMETIS)
    set(ENABLE_PARMETIS OFF)
endif (NOT DEFINED ENABLE_PARMETIS)

# ... Caliper (precise timing information)
if (NOT DEFINED ENABLE_CALIPER)
    set(ENABLE_CALIPER OFF)
endif (NOT DEFINED ENABLE_CALIPER)



# Testing related options, off by default
# ... Build the tests?
if (NOT DEFINED BUILD_TESTS)
    set(BUILD_TESTS OFF)
endif (NOT DEFINED BUILD_TESTS)

# ... Build bookleaf-diff (equivalent to BUILD_TESTS by default)
if (NOT DEFINED BUILD_BOOKLEAF_DIFF)
    set(BUILD_BOOKLEAF_DIFF ${BUILD_TESTS})
endif (NOT DEFINED BUILD_BOOKLEAF_DIFF)

# ... ZLib (used for (de)compressing test dumps), equivalent to BUILD_TESTS by
# ... default
if (NOT DEFINED ENABLE_ZLIB)
    set(ENABLE_ZLIB ${BUILD_TESTS})
endif (NOT DEFINED ENABLE_ZLIB)



# Kokkos defaults
if (NOT DEFINED KOKKOS_ENABLE_OPENMP)
    set(KOKKOS_ENABLE_OPENMP ON)
endif (NOT DEFINED KOKKOS_ENABLE_OPENMP)

if (NOT DEFINED KOKKOS_ENABLE_CUDA)
    set(KOKKOS_ENABLE_CUDA OFF)
endif (NOT DEFINED KOKKOS_ENABLE_CUDA)



# ------------------------------------------------------------------------------
# Get VCS information
# ------------------------------------------------------------------------------
# Get the current working branch
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_definitions("-DBOOKLEAF_GIT_BRANCH=${GIT_BRANCH}")
add_definitions("-DBOOKLEAF_GIT_COMMIT_HASH=${GIT_COMMIT_HASH}")



# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
# Core BookLeaf source files
set(BOOKLEAF_LIB_SRC
    ${PROJECT_SOURCE_DIR}/src/common/data.cpp
    ${PROJECT_SOURCE_DIR}/src/common/data_control.cpp
    ${PROJECT_SOURCE_DIR}/src/common/error.cpp
    ${PROJECT_SOURCE_DIR}/src/common/cmd_args.cpp
    ${PROJECT_SOURCE_DIR}/src/common/config.cpp
    ${PROJECT_SOURCE_DIR}/src/common/runtime.cpp
    ${PROJECT_SOURCE_DIR}/src/common/timer_control.cpp

    ${PROJECT_SOURCE_DIR}/src/packages/time/config.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/time/driver/set.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/time/driver/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/time/driver/advance.cpp

    ${PROJECT_SOURCE_DIR}/src/packages/hydro/config.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/get_acceleration.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/get_energy.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/get_eos.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/get_force.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/get_artificial_viscosity.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/get_state.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/get_dt.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/get_cs2.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/init.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/print.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/driver/set.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/kernel/get_acceleration.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/kernel/get_energy.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/kernel/get_force.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/kernel/get_artificial_viscosity.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/kernel/get_dt.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/kernel/init.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/hydro/kernel/print.cpp

    ${PROJECT_SOURCE_DIR}/src/packages/ale/config.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/driver/get_mesh_status.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/driver/get_flux_volume.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/driver/advect.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/driver/hydro_update.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/driver/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/driver/get_dt.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/kernel/get_mesh_velocity.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/kernel/get_mesh_status.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/kernel/advectors.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/kernel/advect.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/ale/kernel/get_dt.cpp

    ${PROJECT_SOURCE_DIR}/src/packages/setup/indicators.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/mesh_region.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/generate_mesh.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/renumber_mesh.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/transfer_mesh.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/types.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/config.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/driver/set_shape_flags.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/driver/set_thermodynamics.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/driver/set_kinematics.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/driver/set_initial_conditions.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/kernel/set_thermodynamics.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/kernel/set_kinematics.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/kernel/set_flags.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/setup/kernel/shapes.cpp

    ${PROJECT_SOURCE_DIR}/src/packages/init/kernel.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/init/driver.cpp

    ${PROJECT_SOURCE_DIR}/src/packages/io/driver/io_driver.cpp

    ${PROJECT_SOURCE_DIR}/src/packages/check/driver/validate.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/check/driver/tests.cpp
    ${PROJECT_SOURCE_DIR}/src/packages/check/kernel/tests.cpp

    ${PROJECT_SOURCE_DIR}/src/utilities/comms/config.cpp

    ${PROJECT_SOURCE_DIR}/src/utilities/misc/boundary_conditions.cpp
    ${PROJECT_SOURCE_DIR}/src/utilities/misc/average.cpp
    ${PROJECT_SOURCE_DIR}/src/utilities/eos/config.cpp
    ${PROJECT_SOURCE_DIR}/src/utilities/eos/get_eos.cpp
    ${PROJECT_SOURCE_DIR}/src/utilities/geometry/geometry.cpp
    ${PROJECT_SOURCE_DIR}/src/utilities/mix/driver/list.cpp
    ${PROJECT_SOURCE_DIR}/src/utilities/mix/kernel/list.cpp
    ${PROJECT_SOURCE_DIR}/src/utilities/density/get_density.cpp

    ${PROJECT_SOURCE_DIR}/src/infrastructure/io/input_deck.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/io/read.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/io/write.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/init.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/kill.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/solver.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/solver/get_dt.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/solver/lagstep.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/solver/get_acceleration.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/solver/get_energy.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/solver/get_force.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/solver/alestep.cpp
    ${PROJECT_SOURCE_DIR}/src/infrastructure/solver/ale_update.cpp

    ${PROJECT_SOURCE_DIR}/src/packages/io/driver/banner.cpp
)

# Build the core code as a static library, which can be linked into the main
# executable and any unit tests.
add_library(libbookleaf STATIC ${BOOKLEAF_LIB_SRC})
set_target_properties(libbookleaf PROPERTIES OUTPUT_NAME bookleaf)
target_include_directories(libbookleaf PUBLIC include)

# The main bookleaf executable.
add_executable(bookleaf ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(bookleaf PRIVATE libbookleaf)



# ------------------------------------------------------------------------------
# Compilation settings
# ------------------------------------------------------------------------------
target_compile_options(libbookleaf PUBLIC --std=c++11)

# CUDA options
if (KOKKOS_ENABLE_CUDA)
    target_compile_options(libbookleaf PUBLIC --expt-extended-lambda)
    add_definitions("-DBOOKLEAF_KOKKOS_CUDA_SUPPORT")
    set(KOKKOS_ENABLE_CUDA_LAMBDA ON)
    #target_compile_options(libbookleaf PUBLIC -Wno-unused-variable)
    #target_compile_options(libbookleaf PUBLIC -Wno-sign-compare)
endif (KOKKOS_ENABLE_CUDA)

# Generate mesh (this can't be disabled at the moment).
add_definitions("-DBOOKLEAF_MESH_GENERATION")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_definitions("-DBOOKLEAF_COMPILER_GNU")
    target_compile_options(libbookleaf PUBLIC -Wall -Wextra -pedantic)
endif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_definitions("-DBOOKLEAF_COMPILER_INTEL")
    target_compile_options(libbookleaf PUBLIC -Wall -Wextra -pedantic)

    # Intel options copied from Fortran version where applicable
    target_compile_options(libbookleaf PUBLIC
        # CMake flag de-duping causes issues here
        # See: https://gitlab.kitware.com/cmake/cmake/issues/15826
        -fp-model source # Implies -fp-model precise also
        -no-ftz
        -mP2OPT_hlo_loop_unroll=false
        -shared-intel)
endif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_definitions("-DBOOKLEAF_COMPILER_CLANG")
    target_compile_options(libbookleaf PUBLIC -Wall -Wextra -pedantic)
endif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")

if (CMAKE_CXX_COMPILER_ID MATCHES "PGI")
    add_definitions("-DBOOKLEAF_COMPILER_PGI")
endif (CMAKE_CXX_COMPILER_ID MATCHES "PGI")

# Build type dependent options/definitions
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug build")

    add_definitions("-DBOOKLEAF_DEBUG")

    if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        target_compile_options(libbookleaf PUBLIC -traceback)
    endif ()

else ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(libbookleaf PUBLIC -ftree-vectorize)

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        target_compile_options(libbookleaf PUBLIC -xHost)
    endif ()
endif ()



# ------------------------------------------------------------------------------
# Find dependencies
# ------------------------------------------------------------------------------
# Locate yaml-cpp
find_package(YamlCpp REQUIRED)
target_include_directories(libbookleaf PUBLIC ${YamlCpp_INCLUDES})
target_link_libraries(libbookleaf PRIVATE ${YamlCpp_LIBRARIES})

# Check OpenMP support
find_package(OpenMP COMPONENTS CXX REQUIRED)
target_compile_options(libbookleaf PUBLIC ${OpenMP_CXX_FLAGS})
target_link_libraries(libbookleaf PUBLIC OpenMP::OpenMP_CXX)

# Kokkos is included as a submodule, as it functioning correctly is dependent on
# it being built in exactly the same manner as the application that uses it
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(KOKKOS_ENABLE_DEBUG ON)
endif (CMAKE_BUILD_TYPE MATCHES "Debug")

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos
    ${CMAKE_CURRENT_BINARY_DIR}/kokkos)

#target_compile_options(libbookleaf PUBLIC -Wno-unused-parameter)
target_include_directories(libbookleaf PUBLIC ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(libbookleaf PUBLIC kokkos)

# Locate zlib
if (ENABLE_ZLIB)
    find_package(ZLIB REQUIRED)

    add_definitions("-DBOOKLEAF_ZLIB_SUPPORT")
    target_include_directories(libbookleaf PUBLIC ${ZLIB_INCLUDES})
    target_link_libraries(libbookleaf PRIVATE ${ZLIB_LIBRARIES})

    target_sources(libbookleaf PRIVATE
        ${PROJECT_SOURCE_DIR}/src/utilities/debug/zlib_compressor.cpp)
endif (ENABLE_ZLIB)

# Locate Typhon
if (ENABLE_TYPHON)
    set(MPI_CXX_SKIP_MPICXX "true")
    find_package(MPI COMPONENTS CXX REQUIRED)
    find_package(Typhon REQUIRED)

    # Print some information on the MPI distribution
    message(STATUS "MPI_CXX_INCLUDE_DIRS = ${MPI_CXX_INCLUDE_DIRS}")
    message(STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")

    target_include_directories(libbookleaf PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(libbookleaf PUBLIC ${MPI_CXX_LIBRARIES})

    if (MPI_CXX_COMPILE_OPTIONS)
        message(STATUS "MPI_CXX_COMPILE_OPTIONS = ${MPI_CXX_COMPILE_OPTIONS}")
        target_compile_options(libbookleaf PUBLIC "${MPI_CXX_COMPILE_OPTIONS}")
    endif (MPI_CXX_COMPILE_OPTIONS)

    if (MPI_CXX_LINK_FLAGS)
        message(STATUS "MPI_CXX_LINK_FLAGS = ${MPI_CXX_LINK_FLAGS}")
        set_target_properties(libbookleaf
            PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
    endif (MPI_CXX_LINK_FLAGS)

    add_definitions("-DBOOKLEAF_MPI_SUPPORT")
    add_definitions("-DBOOKLEAF_MPI_DT_CONTEXT")
    target_include_directories(libbookleaf PUBLIC ${Typhon_INCLUDES})
    target_link_libraries(libbookleaf PUBLIC ${Typhon_LIBRARIES})

    # Extra files needed for splitting the mesh between processors
    target_sources(libbookleaf PRIVATE
        ${PROJECT_SOURCE_DIR}/src/packages/setup/partition_mesh.cpp
        ${PROJECT_SOURCE_DIR}/src/packages/setup/distribute_mesh.cpp
        ${PROJECT_SOURCE_DIR}/src/utilities/comms/partition.cpp
        ${PROJECT_SOURCE_DIR}/src/utilities/comms/exchange.cpp
        ${PROJECT_SOURCE_DIR}/src/utilities/comms/dt_reduce.cpp)

    # Locate ParMETIS (pointless without Typhon)
    if (ENABLE_PARMETIS)
        find_package(METIS REQUIRED)
        find_package(ParMETIS REQUIRED)

        add_definitions("-DBOOKLEAF_PARMETIS_SUPPORT")
        target_include_directories(libbookleaf PUBLIC ${ParMETIS_INCLUDES})
        target_link_libraries(libbookleaf PRIVATE ${ParMETIS_LIBRARIES})
        target_include_directories(libbookleaf PUBLIC ${METIS_INCLUDES})
        target_link_libraries(libbookleaf PRIVATE ${METIS_LIBRARIES})
    endif (ENABLE_PARMETIS)
endif (ENABLE_TYPHON)

# Locate Silo
if (ENABLE_SILO)
    find_package(Silo REQUIRED)
    find_package(HDF5 REQUIRED)

    add_definitions("-DBOOKLEAF_SILO_SUPPORT")
    target_sources(libbookleaf PRIVATE
        ${PROJECT_SOURCE_DIR}/src/packages/io/driver/silo_io_driver.cpp)

    target_include_directories(libbookleaf PUBLIC ${Silo_INCLUDES})
    target_link_libraries(libbookleaf PRIVATE ${Silo_LIBRARIES})
    target_include_directories(libbookleaf PUBLIC ${HDF5_INCLUDES})
    target_link_libraries(libbookleaf PRIVATE ${HDF5_LIBRARIES})
endif (ENABLE_SILO)

# Locate Caliper
if (ENABLE_CALIPER)
    find_package(Caliper REQUIRED)

    add_definitions("-DBOOKLEAF_CALIPER_SUPPORT")
    target_include_directories(libbookleaf PUBLIC ${Caliper_INCLUDES})
    target_link_libraries(libbookleaf PUBLIC ${CaliperCore_LIBRARIES})
    target_link_libraries(libbookleaf PUBLIC ${CaliperMPI_LIBRARIES})
endif (ENABLE_CALIPER)



# ------------------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------------------
if (NOT KOKKOS_ENABLE_CUDA)
    add_subdirectory(tools)
endif (NOT KOKKOS_ENABLE_CUDA)



# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
include(GNUInstallDirs)

# Just install the executable, as the library is statically linked and not
# useful on its own.
install(
    TARGETS bookleaf
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)



# ------------------------------------------------------------------------------
# Testing (requires zlib for decompressing dumps)
# ------------------------------------------------------------------------------
if (NOT KOKKOS_ENABLE_CUDA)
    if (BUILD_TESTS AND ENABLE_ZLIB)
        enable_testing()
        add_subdirectory(test)
    elseif (BUILD_TESTS AND NOT ENABLE_ZLIB)
        message(WARNING "Not building tests as ZLib was disabled")
    endif ()
endif (NOT KOKKOS_ENABLE_CUDA)
