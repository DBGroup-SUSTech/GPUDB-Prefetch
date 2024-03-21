import matplotlib.pyplot as plt
import numpy as np 
import os
import glob
import re


gp = [[] for _ in range(6)]
amac = [[] for _ in range(6)]
imv = [[] for _ in range(6)]

name = 'A'

# def my(x):
    # return int(x.split('.')[0].split('_')[1])

for file_path in glob.glob(name + '*.res'):
    if file_path[0] == name:
        print(file_path, end=':\n')
        f = open(file_path, "r")
        PDIST = -1+int(file_path.split('.')[0].split('_')[1])
        TEST = file_path.split('.')[0].split('_')[2]
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
        if len(ms) == 3:
            gp[PDIST].append(float(ms[0]))
            amac[PDIST].append(float(ms[1]))
            imv[PDIST].append(float(ms[2]))
            

data_gp = [np.mean(gp[i]) for i in range(6)]
data_amac = [np.mean(amac[i]) for i in range(6)]
data_imv = [np.mean(imv[i]) for i in range(6)]

print("gp ", data_gp)

print("amac", data_amac)


x = range(1, 6+1)

# plt.plot(x, data_gp, label='gp')
# plt.plot(x, data_amac, label='amac')
# plt.plot(x, data_imv, label='imv')

# plt.xlabel('BlockSize')
# plt.ylabel('512MB Running Time (ms)')
# # 添加图例
# plt.legend()

# # 显示图表s
# # plt.show()

# plt.savefig(name + "_new_res.png")


# N=268435456 taskset -c 0 GS=144 BS=128 ./build/btree





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
    zorders = [1,2,3,4,5]
    for legend in source["legends"]:
        # plt.plot(xrange, source['y'][legend], '-'+marker, label=source["legend_names"][legend], mfc="none", markersize=14)
        plt.plot(xrange, [tt/1000000000 for tt in source['y'][legend]], '-'+markers[legend], color=colors[legend], label=source["legend_names"][legend], markersize=marker_size[legend], linewidth=2, zorder=zorders[legend],alpha=1)
    
    plt.xticks(xrange, [x_process(i) for i in source["x"]])
    # plt.xscale("log", base=2)
    # plt.xlabel("Tested range ($\\times10^3$)")
    # plt.xlabel("Number of threads")
    # plt.xlabel("lookup ratio")
    # plt.xlabel("Standard deviation ($\\times 10^4 $)")
    plt.xlabel('Prefetch Distance')
    plt.ylabel('Throughput (Btps)')
    # plt.xscale('log', base=2)
    # plt.ylabel("Avg. active threads per warp")
    # plt.title("Data Num: "+ str(title))
    # plt.yscale('log', base=10)
    plt.ylim(0.2, 0.4)
    plt.legend(loc='lower right', borderaxespad=0, frameon=False, fontsize=19,ncol = 2)
    plt.rcParams['axes.linewidth'] = 2
    plt.savefig(save_path, bbox_inches='tight')
    # plt.plot(x, data["ethereum_hash"],'-kv',mfc="none",markersize=18)
    # plt.plot(x, data["gpu_hash_kernel"], '-ks',mfc='none',markersize=18)


source = {
    "x": [],  # x轴数据，可以在后续添加
    "y": {  # y轴数据，每个图例是一个列表
        # 0: [0] * 5,  # 初始化长度为10的0列表
        0: [0] * 6,
        1: [0] * 6,
        2: [0] * 6,
        # 4: [0] * 5,
    },
    "legends": [0, 1, 2],  # 图例标识，这里直接使用数字0到4
    "legend_names": {  # 图例名称映射
        #  0: "naive",
        0: "GP",
        1: "AMAC",
        2: "IMV",
        # 4: "spp",
    },
}


source["x"] = [str(i) for i in range(1, 6+1)]
print(source["x"])

source["y"][0] = data_gp
source["y"][1] = data_amac
source["y"][2] = data_imv

line_plot(source, 'Figure10(b).pdf', "Prefetch Distance")