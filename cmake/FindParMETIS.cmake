# - Check for the presence of ParMETIS
#
# The following variables are set when ParMETIS is found:
#  ParMETIS_FOUND     = Set to true, if all components of ParMETIS have been found.
#  ParMETIS_INCLUDES  = Include path for the header files of ParMETIS.
#  ParMETIS_LIBRARIES = Link these to use ParMETIS.
#
if (NOT ParMETIS_FOUND)
    if (NOT ParMETIS_ROOT_DIR)
        set (ParMETIS_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
    endif (NOT ParMETIS_ROOT_DIR)

    find_path (ParMETIS_INCLUDES
        NAMES parmetis.h
        HINTS ${ParMETIS_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES include
    )

    find_library (ParMETIS_LIBRARIES parmetis
        HINTS ${ParMETIS_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib
    )

    find_package_handle_standard_args (
        ParMETIS DEFAULT_MSG
        ParMETIS_LIBRARIES
        ParMETIS_INCLUDES
    )

    if (ParMETIS_FOUND)
        if (NOT ParMETIS_FIND_QUIETLY)
            message (STATUS "Found components for ParMETIS")
            message (STATUS "ParMETIS_INCLUDES  = ${ParMETIS_INCLUDES}")
            message (STATUS "ParMETIS_LIBRARIES = ${ParMETIS_LIBRARIES}")
        endif (NOT ParMETIS_FIND_QUIETLY)
    else (ParMETIS_FOUND)
        if (ParMETIS_FIND_REQUIRED)
            message (FATAL_ERROR "Could not find ParMETIS!")
        endif (ParMETIS_FIND_REQUIRED)
    endif (ParMETIS_FOUND)

    mark_as_advanced (
        ParMETIS_ROOT_DIR
        ParMETIS_INCLUDES
        ParMETIS_LIBRARIES
    )

endif (NOT ParMETIS_FOUND)
