#!/usr/bin/env python3
import os

TOTAL_SAMPLES = 1000000     # how many samples in total
CHUNK = 50000               # samples per SLURM job
OUTPUT_DIR = "slurm_jobs"   # directory for .sh files

os.makedirs(OUTPUT_DIR, exist_ok=True)

for offset in range(0, TOTAL_SAMPLES, CHUNK):
    script_name = f"slurm_{offset:06d}.sh"
    script_path = os.path.join(OUTPUT_DIR, script_name)

    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name "gen_data_large_edm_{offset:06d}"
#SBATCH -p proxima
#SBATCH --output edm-%x.%J.%N.out
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node 1
#SBATCH --container ~/pl0467-01/project_data/container/edm.sif

echo "start"

python3 generate_dataset.py \\
    --source_dir /home/mackop/Github/vps_n/sat_data \\
    --source_train_list /home/mackop/Github/vps_n/train_list.txt \\
    --source_val_list /home/mackop/Github/vps_n/val_list.txt \\
    --output_dir /scratch/mackop/generated_data \\
    --num_train {CHUNK} \\
    --num_val 0 \\
    --offset {offset}

echo "done"
""")

    os.chmod(script_path, 0o755)

print(f"Generated SLURM scripts in: {OUTPUT_DIR}")
