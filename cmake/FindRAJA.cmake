# - Check for the presence of RAJA
#
# The following variables are set when RAJA is found:
#  RAJA_FOUND     = Set to true, if all components of RAJA have been found.
#  RAJA_INCLUDES  = Include path for the header files of RAJA.
#  RAJA_LIBRARIES = Link these to use RAJA.
#
if (NOT RAJA_FOUND)
    if (NOT RAJA_ROOT_DIR)
        set (RAJA_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
    endif (NOT RAJA_ROOT_DIR)

    find_path (RAJA_INCLUDES
        NAMES RAJA/RAJA.hpp
        HINTS ${RAJA_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES include
    )

    find_library (RAJA_LIBRARIES RAJA
        HINTS ${RAJA_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib
    )

    find_package_handle_standard_args (
        RAJA DEFAULT_MSG
        RAJA_LIBRARIES
        RAJA_INCLUDES
    )

    if (RAJA_FOUND)
        if (NOT RAJA_FIND_QUIETLY)
            message (STATUS "Found components for RAJA")
            message (STATUS "RAJA_INCLUDES  = ${RAJA_INCLUDES}")
            message (STATUS "RAJA_LIBRARIES = ${RAJA_LIBRARIES}")
        endif (NOT RAJA_FIND_QUIETLY)
    else (RAJA_FOUND)
        if (RAJA_FIND_REQUIRED)
            message (FATAL_ERROR "Could not find RAJA!")
        endif (RAJA_FIND_REQUIRED)
    endif (RAJA_FOUND)

    mark_as_advanced (
        RAJA_ROOT_DIR
        RAJA_INCLUDES
        RAJA_LIBRARIES
    )

endif (NOT RAJA_FOUND)
