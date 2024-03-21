import matplotlib.pyplot as plt
import numpy as np 
import os
import glob
import re

import math
# import pandas as pd


gp = [[] for _ in range(2)]
spp = [[] for _ in range(2)]
amac = [[] for _ in range(2)]
imv = [[] for _ in range(2)]

name = 'Z'

# def my(x):
    # return int(x.split('.')[0].split('_')[1])

for file_path in glob.glob(name + '*.res'):
    if file_path[0] == name:
        print(file_path, end=':\n')
        f = open(file_path, "r")
        shared = bool(file_path.split('.')[0].split('_')[1] == 'shared')
        # test = file_path.split('.')[0].split('_')[2]
        ms = []
        while True:
            a = f.readline()
            tmp = a
            if not a:
                break
            a = a.split(',')
            if len(a) >= 2 and 'probe(S)' in a[0]:
                a = tmp.split('ms')[1].split(' tps ')[0].split(', ')[1]
                # print('\t', a)
                ms.append( float(a) )
        if len(ms) == 5:
            gp[shared].append(float(ms[1]))
            spp[shared].append(float(ms[2]))
            amac[shared].append(float(ms[3]))
            imv[shared].append(float(ms[4]))
            

data_gp = [np.mean(gp[i]) for i in range(2)]
data_spp = [np.mean(spp[i]) for i in range(2)]
data_amac = [np.mean(amac[i]) for i in range(2)]
data_imv = [np.mean(imv[i]) for i in range(2)]


def bar_plot(source, save_path, x_process = lambda x:x):
    width = 0.2
    plt.figure(figsize=(7, 4))
    plt.rc('font', size=20)
    xrange = np.arange(len(source["x"]))
    hatchs = ['', 'xxxxx', '/////', '\\\\\\', 'ooooo', '-----', '+++++'][0:len(source["legends"])]
    for legend, h, i in zip(source["legends"], hatchs, range(0, len(source["legends"]))):
        plt.bar(xrange-(width*len(source["legends"])/2 - width*i - width/2), source['y'][legend], width, edgecolor='k', color='w', hatch=h, label=source["legend_names"][legend])
    
    plt.xticks(xrange, [x_process(i) for i in source["x"]])
    # plt.yticks(fontsize=18)
    plt.legend(loc='upper left', borderaxespad=0, frameon=False,ncol = 2)
    plt.ylabel('Throughput(B tuples/s)')
    plt.ylim(0, 0.5)
    plt.xlabel('Prefetch Algorithm')
    plt.savefig(save_path, bbox_inches='tight')

z = list(zip(data_gp, data_spp, data_amac, data_imv))

A, B = list(z[0]), list(z[1])

import numpy as np

A = list(np.array(A) / (10**9))
B = list(np.array(B) / (10**9))
print(A)
print(B)

data = {
    'x': ["GP", "SPP", "AMAC", "IMV"],
    'y': {
        'A': A,
        'B': B
    },
    'legends': ['A', 'B'],
    'legend_names': {'A': 'in stack', 'B': 'in shared'}
}

# 保存图形
bar_plot(data, 'Figure14.pdf')





