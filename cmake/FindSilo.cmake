# - Check for the presence of Silo
#
# The following variables are set when Silo is found:
#  Silo_FOUND      = Set to true, if all components of Silo have been found.
#  Silo_INCLUDES   = Include path for the header files of Silo
#  Silo_LIBRARIES  = Link these to use Silo
#  Silo_LFLAGS     = Linker flags (optional)
#
if (NOT Silo_FOUND)

    if (NOT Silo_ROOT_DIR)
        set (Silo_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
    endif (NOT Silo_ROOT_DIR)

    find_file (Silo_SETTINGS libsiloh5.settings
        HINTS ${Silo_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib
    )

    find_path (Silo_INCLUDES
        NAMES silo.h pmpio.h
        HINTS ${Silo_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES include
    )

    find_library (Silo_LIBRARIES siloh5
        HINTS ${Silo_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib
    )

    find_program (Silo_SILOFILE_EXECUTABLE silofile
        HINTS ${Silo_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES bin
    )

    find_program (Silo_SILODIFF_EXECUTABLE silodiff
        HINTS ${Silo_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES bin
    )

    find_program (Silo_SILOCK_EXECUTABLE silock
        HINTS ${Silo_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES bin
    )

    find_package_handle_standard_args (
        Silo DEFAULT_MSG
        Silo_LIBRARIES
        Silo_INCLUDES)

    if (Silo_FOUND)
        if (NOT Silo_FIND_QUIETLY)
            message (STATUS "Found components for Silo")
            message (STATUS "Silo_INCLUDES  = ${Silo_INCLUDES}")
            message (STATUS "Silo_LIBRARIES = ${Silo_LIBRARIES}")
        endif (NOT Silo_FIND_QUIETLY)
    else (Silo_FOUND)
        if (Silo_FIND_REQUIRED)
            message (FATAL_ERROR "Could not find Silo!")
        endif (Silo_FIND_REQUIRED)
    endif (Silo_FOUND)

    mark_as_advanced (
        Silo_ROOT_DIR
        Silo_SETTINGS
        Silo_INCLUDES
        Silo_LIBRARIES
    )

endif (NOT Silo_FOUND)
