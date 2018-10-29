# - Check for the presence of CUB
#
# The following variables are set when CUB is found:
#  CUB_FOUND     = Set to true, if all components of CUB have been found.
#  CUB_INCLUDES  = Include path for the header files of CUB.
#
if (NOT CUB_FOUND)
    if (NOT CUB_ROOT_DIR)
        set (CUB_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
    endif (NOT CUB_ROOT_DIR)

    find_path (CUB_INCLUDES
        NAMES cub.cuh
        HINTS ${CUB_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES include/cub
    )

    find_package_handle_standard_args (
        CUB DEFAULT_MSG
        CUB_INCLUDES
    )

    if (CUB_FOUND)
        if (NOT CUB_FIND_QUIETLY)
            message (STATUS "Found components for CUB")
            message (STATUS "CUB_INCLUDES  = ${CUB_INCLUDES}")
        endif (NOT CUB_FIND_QUIETLY)
    else (CUB_FOUND)
        if (CUB_FIND_REQUIRED)
            message (FATAL_ERROR "Could not find CUB!")
        endif (CUB_FIND_REQUIRED)
    endif (CUB_FOUND)

    mark_as_advanced (
        CUB_ROOT_DIR
        CUB_INCLUDES
    )

endif (NOT CUB_FOUND)
