#!/bin/bash

echo "Generating synthetic data."

params_dir='parameters'

make clean &> /dev/null
make syn_gen &> /dev/null
../../bin/syn_gen "$params_dir/parameters_$1" "syn_data.out" &> /dev/null

echo "Synthetic data generated and written to syn_data.out."

make clean &> /dev/null