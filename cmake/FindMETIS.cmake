# - Check for the presence of METIS
#
# The following variables are set when METIS is found:
#  METIS_FOUND     = Set to true, if all components of METIS have been found.
#  METIS_INCLUDES  = Include path for the header files of METIS.
#  METIS_LIBRARIES = Link these to use METIS.
#
if (NOT METIS_FOUND)

    if (NOT METIS_ROOT_DIR)
        set (METIS_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
    endif (NOT METIS_ROOT_DIR)

    find_path (METIS_INCLUDES
        NAMES metis.h
        HINTS ${METIS_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES include
    )

    find_library (METIS_LIBRARIES metis
        HINTS ${METIS_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib
    )

    find_package_handle_standard_args (
        METIS DEFAULT_MSG
        METIS_LIBRARIES
        METIS_INCLUDES
    )

    if (METIS_FOUND)
        if (NOT METIS_FIND_QUIETLY)
            message (STATUS "Found components for METIS")
            message (STATUS "METIS_INCLUDES  = ${METIS_INCLUDES}")
            message (STATUS "METIS_LIBRARIES = ${METIS_LIBRARIES}")
        endif (NOT METIS_FIND_QUIETLY)
    else (METIS_FOUND)
        if (METIS_FIND_REQUIRED)
            message (FATAL_ERROR "Could not find METIS!")
        endif (METIS_FIND_REQUIRED)
    endif (METIS_FOUND)

    mark_as_advanced (
        METIS_ROOT_DIR
        METIS_INCLUDES
        METIS_LIBRARIES
    )

endif (NOT METIS_FOUND)
