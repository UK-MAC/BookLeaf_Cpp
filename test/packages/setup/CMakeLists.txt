add_executable(
    set_flag
    ${CMAKE_CURRENT_SOURCE_DIR}/set_flag.cpp)

add_executable(
    set_flag_if
    ${CMAKE_CURRENT_SOURCE_DIR}/set_flag_if.cpp)

add_executable(
    intersect
    ${CMAKE_CURRENT_SOURCE_DIR}/intersect.cpp)

add_executable(
    subdivide
    ${CMAKE_CURRENT_SOURCE_DIR}/subdivide.cpp)

set(TEST_TARGETS set_flag set_flag_if intersect subdivide)

foreach (TEST_TARGET ${TEST_TARGETS})
    target_link_libraries(${TEST_TARGET} PRIVATE libbookleaf)

    add_test(
        NAME test_${TEST_TARGET}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ./${TEST_TARGET}
    )
endforeach (TEST_TARGET ${TEST_TARGETS})
