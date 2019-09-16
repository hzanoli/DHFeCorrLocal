#!/bin/bash

# SGE options:
#
# Use bash shell
#$ -S /bin/bash
#
# Keep environment variables
#$ -V
#
# Use current working directory
#$ -cwd
#

echo "$3" | alien-token-init "$2"

echo "Downloading $4 to $S5"
python "$1"/io/download_file.py "$4" "$5"

