# naive, different fanout

import os
import subprocess
import re

# Run naive

print(os.getcwd())

for num in range(1, 6+1):
    backUpFileName = 'include/btree/backup_common.cuh'
    fileName = 'include/btree/common.cuh'
    subprocess.run(f"sed 's/COMMON_PDIST [0-9]/COMMON_PDIST {num}/g' {backUpFileName} > {fileName}", shell=True, check=True)

    subprocess.run("cd build/ && taskset -c 0 make", shell=True, check=True)
    # subprocess.run("", shell=True)
    subprocess.run("cd ..", shell=True, check=True)
    for i in range(4):
        subprocess.run(f"N=268435456 GS=144 BS=128 taskset -c 0 ./build/btree --gtest_filter='-*naive*' > A_{num}_{i}_date0313.res", shell=True)

# N=134217728 GS=144 BS=128 taskset -c 0 ./build/btree --gtest_filter='*naive*'