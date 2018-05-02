add_executable(
    init_artificial_viscosity
    ${CMAKE_CURRENT_SOURCE_DIR}/init_artificial_viscosity.cpp)

add_executable(
    limit_artificial_viscosity
    ${CMAKE_CURRENT_SOURCE_DIR}/limit_artificial_viscosity.cpp)

add_executable(
    get_artificial_viscosity
    ${CMAKE_CURRENT_SOURCE_DIR}/get_artificial_viscosity.cpp)

add_executable(
    get_force_pressure
    ${CMAKE_CURRENT_SOURCE_DIR}/get_force_pressure.cpp)

add_executable(
    get_force_viscosity
    ${CMAKE_CURRENT_SOURCE_DIR}/get_force_viscosity.cpp)

add_executable(
    get_energy
    ${CMAKE_CURRENT_SOURCE_DIR}/get_energy.cpp)

add_executable(
    scatter_acceleration
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_acceleration.cpp)

add_executable(
    get_acceleration
    ${CMAKE_CURRENT_SOURCE_DIR}/get_acceleration.cpp)

add_executable(
    apply_acceleration
    ${CMAKE_CURRENT_SOURCE_DIR}/apply_acceleration.cpp)

set(TEST_TARGETS
        init_artificial_viscosity
        limit_artificial_viscosity
        get_artificial_viscosity
        get_force_pressure
        get_force_viscosity
        get_energy
        scatter_acceleration
        get_acceleration
        apply_acceleration)

foreach (TEST_TARGET ${TEST_TARGETS})
    target_link_libraries(${TEST_TARGET} PRIVATE libbookleaf)
    target_link_libraries(${TEST_TARGET} PRIVATE libbookleaf-diff)

    add_test(
        NAME test_${TEST_TARGET}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND
            ./${TEST_TARGET}
            ${CMAKE_CURRENT_SOURCE_DIR}/pre_${TEST_TARGET}.bldump
            ${CMAKE_CURRENT_SOURCE_DIR}/post_${TEST_TARGET}.bldump
    )
endforeach (TEST_TARGET ${TEST_TARGETS})
