#!/bin/bash



# Test whether bookleaf builds successfully in all different configurations en
# masse



# the the script was run from (should be project root)
ROOT_DIR=$(pwd)
if [ ! -f CMakeLists.txt ]; then
    exit 1
fi

# make tmp directory
TMP_DIR=$(mktemp -d)
if [[ ! "$TMP_DIR" || ! -d "$TMP_DIR" ]]; then
    echo "Could not create temp dir"
    exit 1
fi

# deletes the temp directory
function cleanup {
    rm -rf "$TMP_DIR"
}

trap cleanup EXIT

function test_config {
    cd $TMP_DIR

    BUILD_TYPE="$1"
    ENABLE_TYPHON="$2"
    ENABLE_METIS="$3"
    ENABLE_SILO="$4"

    cmake \
        -DCMAKE_INSTALL_PREFIX=$HOME \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DENABLE_TYPHON="$ENABLE_TYPHON" \
        -DENABLE_METIS="$ENABLE_METIS" \
        -DENABLE_SILO="$ENABLE_SILO" \
        "$ROOT_DIR"

    if [ ! $? -eq 0 ]; then
        echo "Failed to configure"
        exit 1
    fi

    make -j8
    if [ ! $? -eq 0 ]; then
        echo "Failed to build"
        exit 1
    fi

    rm -rf *

    cd $ROOT_DIR
}

test_config "Release" "false" "false" "false"
test_config "Release" "false" "false" "true"
#test_config "Release" "false" "true" "false" # Can't have METIS w/o Typhon
#test_config "Release" "false" "true" "true"
test_config "Release" "true" "false" "false"
test_config "Release" "true" "false" "true"
test_config "Release" "true" "true" "false"
test_config "Release" "true" "true" "true"

test_config "Debug" "false" "false" "false"
test_config "Debug" "false" "false" "true"
#test_config "Debug" "false" "true" "false" # Can't have METIS w/o Typhon
#test_config "Debug" "false" "true" "true"
test_config "Debug" "true" "false" "false"
test_config "Debug" "true" "false" "true"
test_config "Debug" "true" "true" "false"
test_config "Debug" "true" "true" "true"

exit 0
