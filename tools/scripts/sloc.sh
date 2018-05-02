#!/bin/bash

headers=$(find include/ -name '*.h')
sources=$(find src/ -name '*.cpp')

in_headers=0
for f in ${headers}; do
    in_file=$(wc -l $f | tr -s ' ' | cut -f1 -d' ')
    in_headers=$((in_headers+in_file))
done

in_sources=0
for f in ${sources}; do
    in_file=$(wc -l $f | tr -s ' ' | cut -f1 -d' ')
    in_sources=$((in_sources+in_file))
done

echo -e "$in_headers\tin headers"
echo -e "$in_sources\tin sources"
echo -e "$((in_headers+in_sources))\ttotal"
