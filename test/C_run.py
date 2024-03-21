# naive, different fanout

import os
import subprocess
import re

# Run naive
GS = 72
for BS in range(64, 512+1, 64):
# for BS in range(256, 256+1, 64):
    break
    backUpFileName = 'include/btree/backup_common.cuh'
    fileName = 'include/btree/common.cuh'
    subprocess.run(f"sed 's/MACRO_BLOCKSIZE 128/MACRO_BLOCKSIZE {BS}/g' {backUpFileName} > {fileName}", shell=True, check=True)

    subprocess.run("cd build/ && taskset -c 0 make", shell=True, check=True)
    subprocess.run("cd ..", shell=True, check=True)
    for i in range(4):
        subprocess.run(f"N=268435456 GS={GS} BS={BS} taskset -c 0 ./build/btree > C_{GS}_{BS}_{i}_date0313.res", shell=True)

        
# for BS in range(64, 512+1, 64):
for BS in [64, 128]:
    backUpFileName = 'include/btree/backup_common.cuh'
    fileName = 'include/btree/common.cuh'
    subprocess.run(f"sed 's/MACRO_BLOCKSIZE 128/MACRO_BLOCKSIZE {BS}/g' {backUpFileName} > {fileName}", shell=True, check=True)

    subprocess.run("cd build/ && taskset -c 0 make", shell=True, check=True)
    subprocess.run("cd ..", shell=True, check=True)
    for i in range(4):
        subprocess.run(f"N=268435456 GS={GS} BS={BS} taskset -c 0 ./build/btree --gtest_filter='*spp*' > Cspp_{GS}_{BS}_{i}_date0314.res", shell=True)

        
