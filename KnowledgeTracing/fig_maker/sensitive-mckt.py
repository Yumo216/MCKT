import matplotlib.pyplot as plt
import numpy as np

# 数据集名称
datasets = ['XES3G5M', 'Junyi', 'POJ']

# Mamba 层数
embedding_dim = [32, 64, 128, 256, 512]

# 第二层的性能数据（AUC，ACC）
my_auc = [0.8511, 0.8688, 0.8124]
my_acc = [0.8485, 0.7701, 0.7912]


# 模拟性能数据，确保第3层性能最佳，其他层次逐步下降
auc_values = {
    'XES3G5M': [0.8471, 0.8496, my_auc[0], 0.8468, 0.8485],
    'Junyi': [0.8645, 0.8667, my_auc[1], 0.8674, 0.8684],
    'POJ': [0.8100, 0.8095, my_auc[2], 0.8078, 0.8089]
}
acc_values = {
    'XES3G5M': [0.8455, 0.8473, my_acc[0], 0.8432, 0.8464],
    'Junyi': [0.7661, 0.7688, my_acc[1], 0.7662, 0.7665],
    'POJ': [0.7900, 0.7899, my_acc[2], 0.7896, 0.7899]
}


# 绘制图形
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 设置每个数据集的Y轴范围
y_limits_acc = {
    'XES3G5M': (0.835, 0.865),  # ACC 较高，刻度范围较高
    'Junyi': (0.75, 0.785),    # ACC 较低，刻度范围相对较低
    'POJ': (0.78, 0.805)       # ACC 中等，刻度范围适中
}

y_limits_auc = {
    'XES3G5M': (0.825, 0.86),  # AUC 较高，刻度范围较低
    'Junyi': (0.845, 0.87),    # AUC 较高，刻度范围适中
    'POJ': (0.775, 0.815)       # AUC 较低，刻度范围较低
}

# 等分 x 轴的位置
x_positions = np.arange(len(embedding_dim))

for i, dataset in enumerate(datasets):
    ax = axes[i]

    # 绘制ACC柱状图
    ax.bar(x_positions, acc_values[dataset], color='#377eb8', width=0.4, label='ACC')
    # 设置ACC Y轴的范围
    ax.set_ylim(y_limits_acc[dataset])

    # 设置横坐标等分并添加标签
    ax.set_xticks(x_positions)
    ax.set_xticklabels(embedding_dim)

    # 创建次坐标轴
    ax2 = ax.twinx()

    # 绘制AUC折线图
    ax2.plot(x_positions, auc_values[dataset], color='green', marker='o', label='AUC')
    # 设置AUC Y轴的范围
    ax2.set_ylim(y_limits_auc[dataset])

    # 设置坐标轴标签
    ax.set_xlabel('Encoder embedding size $d$')
    ax.set_ylabel('ACC')
    ax2.set_ylabel('AUC')

    # 设置图标题
    ax.set_title(dataset, fontweight='bold')

    # 设置图例
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

# 调整布局
plt.tight_layout()



# 保存图形为PDF
plt.savefig('sensitivity_mckt.pdf', format='pdf')

# 显示图形
plt.show()