#!/bin/bash

CONDA_PATH=$HOME/SOFTWARE/anaconda

if [ ! -f ../setup.py ]; then
	echo -e "This script must be run from the \
			 the scripts directory...exiting\n"
	exit 1
fi
echo -e "\n *** SETTING UP flowpy ***\n"

source ./helper_funcs.sh

cd ../

FLOWPY_ROOT=${PWD}
echo -e "Root directory: $FLOWPY_ROOT"


echo -e "Using conda environment: flowpy"

source $CONDA_PATH/etc/profile.d/conda.sh 2>/dev/null

test_cmd conda "Check the anaconda root path, conda not found"

conda env list | grep -w flowpy > /dev/null

if [ $? -eq 0 ]; then
	echo -e "Conda environment already exists."
	CONDA_ENV_EXISTS=1
else
	echo -e "Conda environment doesn't exist. Creating a new one"
	CONDA_ENV_EXISTS=0
fi


if [ $CONDA_ENV_EXISTS -eq 0 ]; then
	conda env create -f scripts/flowpy.yml
else
	conda env update -f scripts/flowpy.yml  --prune
fi
test_return "flowpy environment creation failed" 

conda activate flowpy

PYBIN=$(which python3)
test_cmd $PYBIN

# install flowpy

$PYBIN $FLOWPY_ROOT/setup.py install
test_return "flowpy setup.py failed" 

$PYBIN -c "import flowpy" > /dev/null
test_return "flowpy import test failed" 

conda deactivate

echo -e "### Finished Setup of x3d_post"