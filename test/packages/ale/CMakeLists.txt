add_executable(
    update_basis_el
    ${CMAKE_CURRENT_SOURCE_DIR}/update_basis_el.cpp)

add_executable(
    init_basis_nd
    ${CMAKE_CURRENT_SOURCE_DIR}/init_basis_nd.cpp)

add_executable(
    calc_basis_nd
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_basis_nd.cpp)

add_executable(
    flux_basis_nd
    ${CMAKE_CURRENT_SOURCE_DIR}/flux_basis_nd.cpp)

add_executable(
    mass_basis_nd
    ${CMAKE_CURRENT_SOURCE_DIR}/mass_basis_nd.cpp)

add_executable(
    cut_basis_nd
    ${CMAKE_CURRENT_SOURCE_DIR}/cut_basis_nd.cpp)

add_executable(
    active_nd
    ${CMAKE_CURRENT_SOURCE_DIR}/active_nd.cpp)

add_executable(
    flux_el_vl
    ${CMAKE_CURRENT_SOURCE_DIR}/flux_el_vl.cpp)

add_executable(
    flux_nd_vl
    ${CMAKE_CURRENT_SOURCE_DIR}/flux_nd_vl.cpp)

add_executable(
    update_el
    ${CMAKE_CURRENT_SOURCE_DIR}/update_el.cpp)

add_executable(
    update_nd
    ${CMAKE_CURRENT_SOURCE_DIR}/update_nd.cpp)

add_executable(
    sum_flux
    ${CMAKE_CURRENT_SOURCE_DIR}/sum_flux.cpp)

set(TEST_TARGETS
        update_basis_el
        init_basis_nd
        calc_basis_nd
        flux_basis_nd
        mass_basis_nd
        cut_basis_nd
        active_nd
        flux_el_vl
        flux_nd_vl
        update_el
        update_nd
        sum_flux)

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
