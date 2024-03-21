import os
import glob
import re

for file_path in glob.glob('*.res'):
    if file_path[0] == 'B':
    # if re.match('^f*', file_path):
        print(file_path, end=':\n')
        f = open(file_path, "r")
        ms = []
        while True:
            a = f.readline()
            if not a:
                break
            a = a.split(',')
            if len(a) >= 2 and 'probe(S)' in a[0]:
                a = a[1].strip()
                # print('\t', a)
                ms.append( a.split(' ')[0] )