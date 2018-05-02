# - Check for the presence of Caliper
#
# The following variables are set when Caliper is found:
#  Caliper_FOUND     = Set to true, if all components of Caliper have been found.
#  Caliper_INCLUDES  = Include path for the header files of Caliper.
#  Caliper_LIBRARIES = Link these to use Caliper.
#
if (NOT Caliper_FOUND)
    if (NOT Caliper_ROOT_DIR)
        set (Caliper_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
    endif (NOT Caliper_ROOT_DIR)

    find_path (Caliper_INCLUDES
        NAMES caliper/cali.h
        HINTS ${Caliper_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES include
    )

    find_library (CaliperCore_LIBRARIES
        NAMES caliper
        HINTS ${Caliper_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib
    )

    find_library (CaliperMPI_LIBRARIES
        NAMES caliper-mpi
        HINTS ${Caliper_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib
    )

    find_package_handle_standard_args (
        Caliper DEFAULT_MSG
        CaliperCore_LIBRARIES
        CaliperMPI_LIBRARIES
        Caliper_INCLUDES
    )

    if (Caliper_FOUND)
        if (NOT Caliper_FIND_QUIETLY)
            message (STATUS "Found components for Caliper")
            message (STATUS "Caliper_INCLUDES  = ${Caliper_INCLUDES}")
            message (STATUS "CaliperCore_LIBRARIES = ${CaliperCore_LIBRARIES}")
            message (STATUS "CaliperMPI_LIBRARIES = ${CaliperMPI_LIBRARIES}")
        endif (NOT Caliper_FIND_QUIETLY)
    else (Caliper_FOUND)
        if (Caliper_FIND_REQUIRED)
            message (FATAL_ERROR "Could not find Caliper!")
        endif (Caliper_FIND_REQUIRED)
    endif (Caliper_FOUND)

    mark_as_advanced (
        Caliper_ROOT_DIR
        Caliper_INCLUDES
        CaliperCore_LIBRARIES
        CaliperMPI_LIBRARIES
    )

endif (NOT Caliper_FOUND)
