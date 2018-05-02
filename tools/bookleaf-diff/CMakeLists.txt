if (ENABLE_ZLIB)
    set(LIB_BOOKLEAF_DIFF_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/zlib_decompressor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/data_dump.cpp)

    add_library(libbookleaf-diff STATIC ${LIB_BOOKLEAF_DIFF_SRC})
    set_target_properties(libbookleaf-diff PROPERTIES OUTPUT_NAME bookleaf-diff)
    target_compile_features(libbookleaf-diff PUBLIC cxx_std_11)
    target_include_directories(libbookleaf-diff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

    target_include_directories(libbookleaf-diff PUBLIC ${ZLIB_INCLUDES})
    target_link_libraries(libbookleaf-diff PRIVATE ${ZLIB_LIBRARIES})

    add_executable(bookleaf-diff ${CMAKE_CURRENT_SOURCE_DIR}/src/bookleaf_diff.cpp)
    target_link_libraries(bookleaf-diff PRIVATE libbookleaf-diff)

else ()
    message(WARNING "Not building bookleaf-diff as ZLib was disabled")
endif (ENABLE_ZLIB)
