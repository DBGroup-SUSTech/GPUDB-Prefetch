import matplotlib.pyplot as plt
import numpy as np 
import os
import glob
import re

rg = range(8, 8 + 4 * 9, 4)
idx = {}
c = 0
for i in rg:
    idx[i] = c
    c += 1

baseline = [[] for _ in range(9)]
gp = [[] for _ in range(9)]
spp = [[] for _ in range(3)]
amac = [[] for _ in range(9)]
imv = [[] for _ in range(9)]

name = 'B'

for file_path in glob.glob(name + '*.res'):
    if True:
        print(file_path, end=':\n')
        f = open(file_path, "r")
        o = int(file_path.split('.')[0].split('_')[1])
        if o not in idx.keys():
            continue
        id = idx[o]
        test = file_path.split('.')[0].split('_')[2]
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
        if len(ms) == 4:
            baseline[id].append(ms[0])
            gp[id].append(ms[1])
            amac[id].append(ms[2])
            imv[id].append(ms[3])
        if len(ms) == 1:
            spp[id].append(ms[0])

data_baseline = [np.mean(baseline[i]) for i in range(9)]
data_gp = [np.mean(gp[i]) for i in range(9)]
data_spp = [np.mean(spp[i]) for i in range(3)]
data_amac = [np.mean(amac[i]) for i in range(9)]
data_imv = [np.mean(imv[i]) for i in range(9)]


def line_plot(source, save_path, title = "",x_process = lambda x:x):
    plt.figure(figsize=(7, 4))
    plt.rc('font', size=18)
    xrange = np.arange(len(source["x"]))
    # naive imv amac gp spp
    markers = ['s','o', 'X', 'P', '*']
    marker_size = [10, 14, 10, 10, 17]
    # colors = ['blue', 'orange','gray','yellow', 'purple'] yellow:(0.925,0.9098,0.4274)
    colors = ['gray', (1,0.5098,0.3804), (0.3098,0.741,1.0), (0.4705,0.980,0.3490), 'purple']
    # 显示层次。
    zorders = [1, 2, 3, 4, 5]
    for legend in source["legends"]:
        # plt.plot(xrange, source['y'][legend], '-'+marker, label=source["legend_names"][legend], mfc="none", markersize=14)
        r = []
        if source["legend_names"][legend] == 'SPP':
            r = np.arange(3)
        else:
            r = xrange
    
        plt.plot(r, [tt/1000000000 for tt in source['y'][legend]], '-'+markers[legend], color=colors[legend], label=source["legend_names"][legend], markersize=marker_size[legend], linewidth=2, zorder=zorders[legend],alpha=1)
    
    plt.xticks(xrange, [x_process(i) for i in source["x"]])
    # plt.xscale("log", base=2)
    # plt.xlabel("Tested range ($\\times10^3$)")
    # plt.xlabel("Number of threads")
    # plt.xlabel("lookup ratio")
    # plt.xlabel("Standard deviation ($\\times 10^4 $)")
    plt.xlabel('Node Fanout')
    plt.ylabel('Throughput (Btps)')
    # plt.xscale('log', base=2)
    # plt.ylabel("Avg. active threads per warp")
    # plt.title("Data Num: "+ str(title))
    # plt.yscale('log', base=10)
    plt.ylim(0.1, 0.4)
    plt.legend(loc='lower right', borderaxespad=0, frameon=False, fontsize=19,ncol = 2)
    plt.rcParams['axes.linewidth'] = 2
    plt.savefig(save_path, bbox_inches='tight')
    # plt.plot(x, data["ethereum_hash"],'-kv',mfc="none",markersize=18)
    # plt.plot(x, data["gpu_hash_kernel"], '-ks',mfc='none',markersize=18)


source = {
    "x": [],  # x轴数据，可以在后续添加
    "y": {  # y轴数据，每个图例是一个列表
        # 0: [0] * 5,  # 初始化长度为10的0列表
        0: [0] * 9,
        1: [0] * 9,
        2: [0] * 9,
        3: [0] * 9,
        4: [0] * 9
    },
    "legends": [0, 1, 2, 3, 4],  # 图例标识，这里直接使用数字0到4
    "legend_names": {  # 图例名称映射
        0: "baseline",
        1: "GP",
        2: "SPP",
        3: "AMAC",
        4: "IMV"
    },
}

source["x"] = [str(i) for i in rg]
print(source["x"])

source["y"][0] = data_baseline
source["y"][1] = data_gp
source["y"][2] = data_spp
source["y"][3] = data_amac
source["y"][4] = data_imv

line_plot(source, 'Figure13.pdf', "Node Fanout")