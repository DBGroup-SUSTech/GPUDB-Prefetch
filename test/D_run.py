# naive, different fanout

import os
import subprocess
import re

# Run naive
BS = 128

backUpFileName = 'include/btree/backup_common.cuh'
fileName = 'include/btree/common.cuh'
subprocess.run(f"sed 's/MACRO_BLOCKSIZE 128/MACRO_BLOCKSIZE {BS}/g' {backUpFileName} > {fileName}", shell=True, check=True)
subprocess.run("cd build/ && make", shell=True, check=True)
subprocess.run("cd ..", shell=True, check=True)

for GS in range(36, 36*8+1, 36):
    subprocess.run(f"N=268435456 GS={GS} BS={BS} ./build/btree > D_{GS}_{BS}.res", shell=True)
