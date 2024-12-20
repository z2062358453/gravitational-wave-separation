# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# overlap_ranges=np.load('./Saparate_task_signal_end2end_RNN/overlap_ranges.npy',allow_pickle=True).item()
# labels = list(overlap_ranges.keys())
# counts_overlap1 = [count[0] for count in overlap_ranges.values()]
# counts_overlap2 = [count[1] for count in overlap_ranges.values()]
# total_counts_overlap1=sum(counts_overlap1)
# total_counts_overlap2=sum(counts_overlap2)
# overlap_percentage1 = [(count / total_counts_overlap1) * 100 for count in counts_overlap1]
# overlap_percentage2 = [(count / total_counts_overlap2) * 100 for count in counts_overlap2]
# x = np.arange(len(labels))  # 标签位置
# width = 0.35  # 柱形宽度
#
# fig, ax = plt.subplots()
# bars1 = ax.bar(x - width / 2, overlap_percentage1, width, label='OverlapA')
# bars2 = ax.bar(x + width / 2, overlap_percentage2, width, label='OverlapB')
#
# ax.set_xlabel('Overlap Ranges')
# ax.set_ylabel('Percentage (%)')
# ax.set_title('Overlap Range Distribution')
# ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation=45)
# ax.legend()
#
# plt.tight_layout()
# plt.savefig('./overlap_histogram.png')
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['mathtext.fontset']='custom'
matplotlib.rcParams['mathtext.rm']='Times New Roman'
# 自定义蓝紫色渐变色图
blue_purple_cmap = LinearSegmentedColormap.from_list('blue_purple_cmap', ['#d1e6fa', '#4b0082'])

# 定义一个函数来绘制热力图子图
def plot_overlap(ax, overlap_file, title):
    # 加载 overlap 结果
    with open(overlap_file, 'rb') as f:
        overlap_results = pickle.load(f)

    # 提取数据
    mass_A = []
    mass_B = []
    overlap_values = []

    for (massa, massb), overlap1 in overlap_results.items():
        mass_A.append(massa)
        mass_B.append(massb)
        overlap_values.append(overlap1)

    # 将数据放入二维网格
    grid_size = 71  # 这里设置为71以匹配数据的维度
    A_bins = np.linspace(10, 80, grid_size)
    B_bins = np.linspace(10, 80, grid_size)

    # 创建二维直方图
    H, _, _ = np.histogram2d(mass_B, mass_A, bins=(B_bins, A_bins), weights=overlap_values)

    # 掩蔽上三角区域
    mask = np.tri(H.shape[0], H.shape[1], k=0)  # 下三角为True，上三角为False
    H_masked = np.ma.array(H.T, mask=mask)

    # 绘制统计图
    c = ax.imshow(H_masked, origin='lower', cmap=blue_purple_cmap, extent=[10, 80, 10, 80], vmin=0, vmax=1)
    # ax.set_xlabel('Chirp Mass B',fontsize=15,fontname='Times New Roman')
    # ax.set_xlabel(r'Chirp Mass B ($M_{\odot}$)', fontsize=15, fontname='Times New Roman')
    ax.set_xlabel(r'$\mathrm{Chirp\ Mass\ B\ (M_{\odot})}$', fontsize=17, fontname='Times New Roman')
    # ax.set_ylabel('Chirp Mass A',fontsize=15,fontname='Times New Roman')
    ax.set_ylabel(r'$\mathrm{Chirp\ Mass\ A\ (M_{\odot})}$', fontsize=17, fontname='Times New Roman')
    ax.set_title(title,fontsize=20, fontname='Times New Roman')
    ax.tick_params(axis='x', labelsize=15)  # Change 10 to your desired font size
    ax.tick_params(axis='y', labelsize=15)  # Change 10 to your desired font size
    # 添加颜色条
    cbar = plt.colorbar(c, ax=ax, ticks=np.linspace(0, 1, num=11))
    cbar.ax.set_yticklabels([f'{i/10:.1f}' for i in range(11)])
    cbar.ax.tick_params(labelsize=15)

# 定义一个函数来绘制柱形图子图
def plot_overlap_histogram(ax, file_path):
    # 加载 overlap_ranges 数据
    overlap_ranges = np.load(file_path, allow_pickle=True).item()

    labels = list(overlap_ranges.keys())
    counts_overlap1 = [count[0] for count in overlap_ranges.values()]
    counts_overlap2 = [count[1] for count in overlap_ranges.values()]

    # 计算总计数
    total_counts_overlap1 = sum(counts_overlap1)
    total_counts_overlap2 = sum(counts_overlap2)
    # total_counts_overlap2 = counts_overlap2[0]
    print(total_counts_overlap1)
    print(total_counts_overlap2)
    # 计算百分比
    percentages_overlap1 = [(count / total_counts_overlap1) * 100 for count in counts_overlap1]
    percentages_overlap2 = [(count / total_counts_overlap2) * 100 for count in counts_overlap2]
    # labels.reverse()
    # percentages_overlap1.reverse()
    # percentages_overlap2.reverse()
    x = np.arange(len(labels))  # 标签位置
    width = 0.35  # 柱形宽度

    bars1 = ax.bar(x - width/2, percentages_overlap1, width, label='Overlap of Signal A')
    bars2 = ax.bar(x + width/2, percentages_overlap2, width, label='Overlap of Signal B')

    ax.set_xlabel('Overlap Ranges',fontsize=17,fontname='Times New Roman')
    ax.set_ylabel('Percentage (%)',fontsize=17,fontname='Times New Roman')
    ax.set_title('(c)',fontsize=20,fontname='Times New Roman')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend(prop={"family": "Times New Roman", "size": 15})
    ax.tick_params(axis='x', labelsize=15)  # Change 10 to your desired font size
    ax.tick_params(axis='y', labelsize=15)  # Change 10 to your desired font size
    # plt.tight_layout(rect=[0,0,0.5,0.95])
    # overlap_ranges=np.load('./Saparate_task_signal_end2end_RNN/overlap_ranges.npy',allow_pickle=True).item()
    # labels = list(overlap_ranges.keys())
    # counts_overlap1 = [count[0] for count in overlap_ranges.values()]
    # counts_overlap2 = [count[1] for count in overlap_ranges.values()]
    # total_counts_overlap1=sum(counts_overlap1)
    # total_counts_overlap2=sum(counts_overlap2)
    # overlap_percentage1 = [(count / total_counts_overlap1) * 100 for count in counts_overlap1]
    # overlap_percentage2 = [(count / total_counts_overlap2) * 100 for count in counts_overlap2]
    # x = np.arange(len(labels))  # 标签位置
    # width = 0.35  # 柱形宽度
    #
    # fig, ax = plt.subplots()
    # bars1 = ax.bar(x - width / 2, overlap_percentage1, width, label='OverlapA')
    # bars2 = ax.bar(x + width / 2, overlap_percentage2, width, label='OverlapB')
    #
    # ax.set_xlabel('Overlap Ranges')
    # ax.set_ylabel('Percentage (%)')
    # ax.set_title('Overlap Range Distribution')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=45)
    # ax.legend()
    #
    # plt.tight_layout()
    # plt.savefig('./overlap_histogram.png')
    # plt.show()
# def plot_overlap_histogram(ax, file_path):
#     # 加载 overlap_ranges 数据
#     overlap_ranges = np.load(file_path, allow_pickle=True).item()
#
#     # 提取区间标签和对应的 overlap1 和 overlap2 计数
#     labels = list(overlap_ranges.keys())
#     counts_overlap1 = [count[0] for count in overlap_ranges.values()]  # overlap1 的计数
#     counts_overlap2 = [count[1] for count in overlap_ranges.values()]  # overlap2 的计数
#
#     # 以 overlap>=0 的区间作为总样本数
#     total_counts_overlap1 = counts_overlap1[0]  # overlap1 的总数（对应 overlap >= 0）
#     total_counts_overlap2 = counts_overlap2[0]  # overlap2 的总数（对应 overlap >= 0）
#
#     # 如果总计数为 0，防止除以 0
#     if total_counts_overlap1 == 0:
#         total_counts_overlap1 = 1
#     if total_counts_overlap2 == 0:
#         total_counts_overlap2 = 1
#
#     # 计算百分比
#     percentages_overlap1 = [(count / total_counts_overlap1) * 100 for count in counts_overlap1]
#     percentages_overlap2 = [(count / total_counts_overlap2) * 100 for count in counts_overlap2]
#     labels.reverse()
#     percentages_overlap1.reverse()
#     percentages_overlap2.reverse()
#     # 定义 X 轴位置
#     x = np.arange(len(labels))  # 标签位置
#
#     # 绘制两条折线图
#     ax.plot(x, percentages_overlap1, marker='o', label='Signal A', linestyle='-', color='b')
#     ax.plot(x, percentages_overlap2, marker='s', label='Signal B', linestyle='--', color='r')
#
#     # 设置标签、标题和刻度
#     ax.set_xlabel('Maximum Mismatch', fontsize=15)
#     ax.set_ylabel('Percentage (%)', fontsize=15)
#     ax.set_title('(c)', fontsize=18)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=45)
#     ax.set_ylim(80, 103)
#     ax.set_yticks(np.arange(80, 101, 5))
#     # 显示图例
#     ax.legend(fontsize=15)
#
#     # 调整 X 和 Y 轴刻度字体大小
#     ax.tick_params(axis='x', labelsize=15)
#     ax.tick_params(axis='y', labelsize=15)

# 创建包含三个子图的图形
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 7))

# 绘制第一个子图（从 overlap_a.pkl 加载的数据）
plot_overlap(ax1, './overlap_a.pkl', '(a)')

# 绘制第二个子图（从 overlap_b.pkl 加载的数据）
plot_overlap(ax2, './overlap_b.pkl', '(b)')

# 绘制第三个子图（柱形图）
plot_overlap_histogram(ax3, './Saparate_task_signal_end2end_RNN/overlap_ranges.npy')
# fig.patch.set_facecolor('white')  # 设置整个图形的背景颜色
# ax1.set_facecolor('white')        # 设置子图的背景颜色
# ax2.set_facecolor('white')
# ax3.set_facecolor('white')

# 调整布局以避免子图重叠
plt.tight_layout()
plt.savefig('./overlap_ranges.tif')
# 显示图形
plt.show()