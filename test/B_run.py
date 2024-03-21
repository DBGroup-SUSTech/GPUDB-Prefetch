# naive, different fanout

import os
import subprocess
import re

# Run naive

range1 = [8, 12]
range2 = [16, 20, 24, 28, 32, 36, 40]
for num in range1: # contain spp
    break
    print("num = ", num)
    backUpFileName = 'include/btree/backup_common.cuh'
    fileName = 'include/btree/common.cuh'
    subprocess.run(f"sed 's/MACRO_MAX_ENTRIES 14/MACRO_MAX_ENTRIES {num}/g' {backUpFileName} > {fileName}", shell=True, check=True)

    subprocess.run("cd build/ && taskset -c 0 make", shell=True, check=True)
    subprocess.run("cd ..", shell=True, check=True)
    for i in range(4):
        subprocess.run(f"N=268435456 GS=144 BS=128 taskset -c 0 ./build/btree > B_{num}_{i}_date0314.res", shell=True)

for num in range2:
    backUpFileName = 'include/btree/backup_common.cuh'
    fileName = 'include/btree/common.cuh'
    subprocess.run(f"sed 's/MACRO_MAX_ENTRIES [0-9][0-9]/MACRO_MAX_ENTRIES {num}/g' {backUpFileName} > {fileName}", shell=True, check=True)

    subprocess.run("cd build/ && taskset -c 0 make", shell=True, check=True)
    subprocess.run("cd ..", shell=True, check=True)
    for i in range(4):
         subprocess.run(f"N=268435456 GS=144 BS=128 taskset -c 0 ./build/btree > B_{num}_{i}_date0314.res", shell=True)

# N=134217728 GS=144 BS=128 taskset -c 0 ./build/btree --gtest_filter='*naive*'
        
# 18