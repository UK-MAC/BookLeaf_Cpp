#!/bin/bash

in_pattern="$1"
out_pattern="$2"

headers=$(find include/ -name '*.h')
sources=$(find src/ -name '*.cpp')

for f in ${headers}; do
    sed -i "s/${in_pattern}/${out_pattern}/g" ${f}
done

for f in ${sources}; do
    sed -i "s/${in_pattern}/${out_pattern}/g" ${f}
done
