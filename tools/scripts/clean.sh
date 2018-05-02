#!/bin/bash

rm -f *.dat*
rm -f *.silo

if [ -d initial_dump ]; then
    rm -r initial_dump
fi

if [ -d final_dump ]; then
    rm -r final_dump
fi
