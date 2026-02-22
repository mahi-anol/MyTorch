#!/bin/bash

module0_dir="minitorch-module0"
sub_dirs=("minitorch" "test" "project")

minitorch_module_files=("__init__.py" "operators.py" "module.py" "testing.py" "datasets.py")
test_files=("conftest.py" "test_operators.py" "test_module.py")
base_files=("requirements.txt" "pyproject.toml")
# create directories
for sub in "${sub_dirs[@]}";do
    if [ -d "$module0_dir/$sub" ]; then
        echo "$module0_dir/$sub Already Exists"
    else
        mkdir -p "$module0_dir/$sub"
    fi
done

# create minitorch module files.
for file in "${minitorch_module_files[@]}";do
    if [ ! -f "$module0_dir/${sub_dirs[0]}/$file" ]; then
        touch "$module0_dir/${sub_dirs[0]}/$file"
    else
        echo "$module0_dir/${sub_dirs[0]}/${file} already exists." 
    fi
done

#Creating test files
for file in "${test_files[@]}";do
    if [ ! -f "$module0_dir/${sub_dirs[1]}/$file" ]; then
        touch "$module0_dir/${sub_dirs[1]}/$file"
    else
        echo "$module0_dir/${sub_dirs[1]}/${file} already exists." 
    fi
done

#Creating base files
for file in "${base_files[@]}";do
    if [ ! -f "$module0_dir/$file" ]; then
        touch "$module0_dir/$file"
    else
        echo "$module0_dir/${file} already exists." 
    fi
done