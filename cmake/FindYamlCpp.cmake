# - Check for the presence of yaml-cpp
#
# The following variables are set when yaml-cpp is found:
#  YamlCpp_FOUND     = Set to true, if all components of yaml-cpp have been found.
#  YamlCpp_INCLUDES  = Include path for the header files of yaml-cpp.
#  YamlCpp_LIBRARIES = Link these to use yaml-cpp.
#
if (NOT YamlCpp_FOUND)
    if (NOT YamlCpp_ROOT_DIR)
        set (YamlCpp_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
    endif (NOT YamlCpp_ROOT_DIR)

    find_path (YamlCpp_INCLUDES
        NAMES yaml-cpp/yaml.h
        HINTS ${YamlCpp_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES include
    )

    find_library (YamlCpp_LIBRARIES yaml-cpp
        HINTS ${YamlCpp_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES lib
    )

    find_package_handle_standard_args (
        YamlCpp DEFAULT_MSG
        YamlCpp_LIBRARIES
        YamlCpp_INCLUDES
    )

if (YamlCpp_FOUND)
    if (NOT YamlCpp_FIND_QUIETLY)
            message (STATUS "Found components for yaml-cpp")
            message (STATUS "YamlCpp_INCLUDES  = ${YamlCpp_INCLUDES}")
            message (STATUS "YamlCpp_LIBRARIES = ${YamlCpp_LIBRARIES}")
        endif (NOT YamlCpp_FIND_QUIETLY)
    else (YamlCpp_FOUND)
        if (YamlCpp_FIND_REQUIRED)
            message (FATAL_ERROR "Could not find yaml-cpp!")
        endif (YamlCpp_FIND_REQUIRED)
    endif (YamlCpp_FOUND)

    mark_as_advanced (
        YamlCpp_ROOT_DIR
        YamlCpp_INCLUDES
        YamlCpp_LIBRARIES
    )

endif (NOT YamlCpp_FOUND)
