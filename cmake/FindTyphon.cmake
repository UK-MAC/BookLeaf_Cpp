# - Check for the presence of Typhon
#
# The following variables are set when Typhon is found:
#  Typhon_FOUND     = Set to true, if all components of Typhon have been found.
#  Typhon_INCLUDES  = Include path for the header files of Typhon.
#  Typhon_LIBRARIES = Link these to use Typhon.
#
if (NOT Typhon_FOUND)
    if (NOT Typhon_ROOT_DIR)
        set (Typhon_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
    endif (NOT Typhon_ROOT_DIR)

    find_path (Typhon_INCLUDES
        NAMES typhon.h
        HINTS ${Typhon_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES include
    )

    find_library (Typhon_LIBRARIES typhon
        HINTS ${Typhon_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib
    )

    find_package_handle_standard_args (
        Typhon DEFAULT_MSG
        Typhon_LIBRARIES
        Typhon_INCLUDES
    )

    if (Typhon_FOUND)
        if (NOT Typhon_FIND_QUIETLY)
            message (STATUS "Found components for Typhon")
            message (STATUS "Typhon_INCLUDES  = ${Typhon_INCLUDES}")
            message (STATUS "Typhon_LIBRARIES = ${Typhon_LIBRARIES}")
        endif (NOT Typhon_FIND_QUIETLY)
    else (Typhon_FOUND)
        if (Typhon_FIND_REQUIRED)
            message (FATAL_ERROR "Could not find Typhon!")
        endif (Typhon_FIND_REQUIRED)
    endif (Typhon_FOUND)

    mark_as_advanced (
        Typhon_ROOT_DIR
        Typhon_INCLUDES
        Typhon_LIBRARIES
    )

endif (NOT Typhon_FOUND)
