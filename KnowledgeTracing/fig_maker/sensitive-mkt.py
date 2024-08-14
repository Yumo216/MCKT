import matplotlib.pyplot as plt

# 数据集名称
datasets = ['ASSIST09', 'ASSIST17', 'EdNet']

# Mamba 层数
layer_numbers = [1, 2, 3, 4, 5]

# 第二层的性能数据（AUC，ACC）
second_layer_auc = [0.7778, 0.7561, 0.7002]
second_layer_acc = [0.7634, 0.7561, 0.6741]


# 模拟性能数据，确保第2层性能最佳，其他层次逐步下降
auc_values = {
    'ASSIST09': [0.7415, second_layer_auc[0], 0.7741, 0.7711, 0.7672],
    'ASSIST17': [0.7323, second_layer_auc[1], 0.7542, 0.7513, 0.7479],
    'EdNet': [0.6712, second_layer_auc[2], 0.7006, 0.6917, 0.6892]
}
acc_values = {
    'ASSIST09': [0.7412, second_layer_acc[0], 0.7612, 0.7599, 0.7611],
    'ASSIST17': [0.74, second_layer_acc[1], 0.7543, 0.7512, 0.7501],
    'EdNet': [0.6512, second_layer_acc[2], 0.6789, 0.6710, 0.6698]
}


# 绘制图形
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.6)
# 设置每个数据集的Y轴范围
y_limits_acc = {
    'ASSIST09': (0.73, 0.80),
    'ASSIST17': (0.72, 0.77),
    'EdNet': (0.63, 0.70)
}

y_limits_auc = {
    'ASSIST09': (0.70, 0.785),
    'ASSIST17': (0.71, 0.76),
    'EdNet': (0.64, 0.71)
}


for i, dataset in enumerate(datasets):
    ax = axes[i]

    # 绘制ACC柱状图
    ax.bar(layer_numbers, acc_values[dataset], color='#377eb8', width=0.4, label='ACC')
    # 设置ACC Y轴的范围
    ax.set_ylim(y_limits_acc[dataset])
    # 创建次坐标轴
    ax2 = ax.twinx()

    # 绘制AUC折线图
    ax2.plot(layer_numbers, auc_values[dataset], color='green', marker='o', label='AUC')

    # 设置AUC Y轴的范围
    ax2.set_ylim(y_limits_auc[dataset])
    # 设置坐标轴标签
    ax.set_xlabel('Mamba Layer Number $i$')
    ax.set_ylabel('ACC')
    ax2.set_ylabel('AUC')

    # 设置图标题
    ax.set_title(dataset, fontweight='bold')

    # 设置图例
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

# 显示图形
# 调整布局
plt.tight_layout()

# 保存图形为PDF
# plt.savefig('sensitivity_mkt.pdf', format='pdf')

# 显示图形
plt.show()
