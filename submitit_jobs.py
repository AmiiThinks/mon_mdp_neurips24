# Use this script to run multiple Hydra sweeps in parallel with SLURM + Joblib.
# Because runs are very short, it's best to run them in parallel over multiple CPUs
# with Hydra Joblib plugin.
# To further parallelize everything, we recommend to submit multiple SLURM jobs
# with low compute requirements. Having low requirements ensures your job gets
# high priority.
# Each job will take care of a chunk of the seed range.
# Note that this works because runs are quick and need only CPUs.

# This script submits 10 jobs, each running a sweep (passed as argument) over different seeds.
# Each job keeps launching N runs in parallel (where N is the numper of CPUs requested).
# Time and memory needed for a run are adjusted depending manually on the environment.

# To check the job's progress
# cat /scratch/USERNAME/slurm_out/JOB_ID_0_log.out

import os
import submitit
import numpy as np
import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--sweeper")
args = parser.parse_args()

username = os.environ["USER"]

n_chunks = 30
seeds_chunks = np.array_split(range(0, 30), n_chunks)

envs = [
    "quicksand_distract",
    "small_loop",
    "two_room_quicksand",
    "two_room_distract_middle",
    "straight",
    "corridor",
    "medium_distract",
    "river_swim",
]
mons = [
    "full",
    "battery",
    "random_nonzero",
    "binary_stateless",
    "n",
    "level",
    "button",
]

for env, mon in list(itertools.product(envs, mons)):
    for seeds in seeds_chunks:
        executor = submitit.AutoExecutor(
            folder=f"/scratch/{username}/slurm_out"
        )  # you will find error logs here

        mem_gb = 4
        if mon in ["battery", "n", "level"]:
            mem_gb = 16

        timeout = 20
        if env in ["river_swim", "two_room_distract_middle", "straight"]:
            timeout = 60
        if mon in ["n", "level", "battery"]:
            timeout *= 3

        executor.update_parameters(
            slurm_account="rrg-mbowling-ad",
            timeout_min=timeout,
            nodes=1,
            cpus_per_task=4,
            mem_gb=mem_gb,
        )

        cmd = (
            "python main.py "
            "-m hydra/launcher=joblib "
            f"hydra/sweeper={args.sweeper} "
            "hydra.launcher.verbose=1000 "
            f"environment={env} "
            f"monitor={mon} "
            "experiment.rng_seed=" + ",".join(map(str, seeds))
        )

        job = executor.submit(os.system, cmd)
        print(f"Submitted job: {job.job_id}")
