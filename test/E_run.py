import os
import subprocess
import re

subprocess.run("cd build/ && taskset -c 0 make", shell=True, check=True)
subprocess.run("cd ..", shell=True, check=True)
# Run naive
nums = [2 ** i for i in range(24, 28+1)]
GS = 72 * 2
BS = 128
for num in nums:
    # backUpFileName = 'include/btree/backup_common.cuh'
    # fileName = 'include/btree/common.cuh'
    # subprocess.run(f"sed 's/MACRO_BLOCKSIZE [0-9][0-9]/MACRO_BLOCKSIZE {BS}/g' {backUpFileName} > {fileName}", shell=True, check=True)
    for k in range(3):
        subprocess.run(f"N={num} GS={GS} BS={BS} taskset -c 0 ./build/btree > E_{num}_{k}.res", shell=True)
