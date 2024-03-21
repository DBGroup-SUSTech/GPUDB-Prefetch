# naive, different fanout

import os
import subprocess
import re

# Run naive

for num in range(12, 32+1, 4):
    break
    print("num = ", num)
    backUpFileName = 'include/btree/backup_common.cuh'
    fileName = 'include/btree/common.cuh'
    subprocess.run(f"sed 's/COMMON_LPW 8/COMMON_LPW {num}/g' {backUpFileName} > {fileName}", shell=True, check=True)

    subprocess.run("cd build/ && taskset -c 0 make", shell=True, check=True)
    subprocess.run("cd ..", shell=True, check=True)
    for i in range(4):
        subprocess.run(f"N=268435456 GS=144 BS=128 taskset -c 0 ./build/btree --gtest_filter='-*naive*' > F_{num}_{i}_date0313.res", shell=True)

for num in range(12,32+1, 4):
    print(f"num = {num} compiling")
    backUpFileName = 'include/btree/backup_common.cuh'
    fileName = 'include/btree/common.cuh'
    subprocess.run(f"sed 's/COMMON_LPW 8/COMMON_LPW {num}/g' {backUpFileName} > {fileName}", shell=True, check=True)

    subprocess.run("cd build/ && taskset -c 0 make", shell=True, check=True)
    subprocess.run("cd ..", shell=True, check=True)
    for i in range(4):
        print (f"num = {num} case = {i}")
        subprocess.run(f"N=268435456 GS=144 BS=128 taskset -c 0 ./build/btree --gtest_filter='*naive*' > Fnaive_{num}_{i}_date0317.res", shell=True)
