#!/bin/bash
# run_empire_job.sh
# Job launcher for Empire model (single run, no parameter loop)

# SGE options
#$ -V                       # Export environment variables
#$ -cwd                     # Run job from current working dir
#$ -N empire_run            # Job name
#$ -o ./hpc_output/         # Stdout log
#$ -e ./hpc_output/         # Stderr log
#$ -l h_rt=12:00:00         # Max runtime
#$ -l mem_free=150G         # Memory
#$ -pe smp 8                # Number of cores
#$ -l hostname="compute-6-2|compute-6-1|"

mkdir -p ./hpc_output/

echo "Starting Empire run on $(hostname)"

# Activate your environment if needed
# module load anaconda3
# source activate empire-env

# Run the Python script
python scripts/run.py -d europe_v51 -c config/run.yaml --force

echo "Empire run completed!"
