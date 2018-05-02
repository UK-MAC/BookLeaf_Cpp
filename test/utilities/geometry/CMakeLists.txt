add_executable(
    get_centroid
    ${CMAKE_CURRENT_SOURCE_DIR}/get_centroid.cpp)

set(TEST_TARGETS get_centroid)

foreach (TEST_TARGET ${TEST_TARGETS})
    target_link_libraries(${TEST_TARGET} PRIVATE libbookleaf)

    add_test(
        NAME test_${TEST_TARGET}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ./${TEST_TARGET}
    )
endforeach (TEST_TARGET ${TEST_TARGETS})
