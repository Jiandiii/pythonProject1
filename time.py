import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 数据准备（可替换为你的真实数据） ----------------------
# 方法名称（x轴标签）
methods = ['TRINA', 'PRINGLE', 'RNCGLN', 'GCN']
# 四个数据集对应的推理时间（每个方法对应4个值，顺序：Cora, CiteSeer, Photo, PubMed）
# 这里是根据原图的近似模拟值，你可以替换成自己的实验数据
inference_times = {
    'Cora': [0.29, 0.06, 0.03, 0.02],
    'CiteSeer': [0.44, 0.06, 0.03, 0.03],
    'Photo': [1.00, 0.36, 0.09, 0.05],
    'PubMed': [3.03, 0.23, 0.52, 0.05]
}
datasets = list(inference_times.keys())
n_methods = len(methods)
n_datasets = len(datasets)

# ---------------------- 2. 绘图参数设置 ----------------------
bar_width = 0.1  # 单个柱子的宽度
# 设置每组起始位置，使组间间距为一个 bar 的宽度 (4个bar + 1个间距 = 5份 bar_width)
x = np.arange(n_methods) * (n_datasets + 1) * bar_width 
# 每个数据集的填充样式（匹配原图的视觉区分）
hatch_styles = ['', '///', '...', '\\\\']  # Cora(空白), CiteSeer(斜线), Photo(点), PubMed(交叉线)
colors = ['white'] * n_datasets  # 柱子背景为白色，靠填充样式区分

# ---------------------- 3. 绘制分组柱状图 ----------------------
plt.figure(figsize=(4, 3))  # 缩小图的尺寸，原来是 (8, 6) 或 (10, 6)

for i in range(n_datasets):
    # 每个数据集的柱子位置 = 基准x + 宽度偏移
    plt.bar(
        x + i * bar_width, 
        inference_times[datasets[i]], 
        width=bar_width, 
        color=colors[i], 
        hatch=hatch_styles[i], 
        edgecolor='black',  # 柱子加黑色边框
        label=datasets[i]
    )

# ---------------------- 4. 图表美化与标签 ----------------------
plt.title('Training Time Comparison', fontsize=12)
plt.ylabel('Training Time (s)', fontsize=12)
# plt.xlabel('Methods', fontsize=12)
plt.xticks(x + bar_width * (n_datasets - 1) / 2, methods, fontsize=10)  # x轴标签居中
plt.yscale('log')  # 纵轴设置为对数刻度（匹配原图）
plt.ylim(1e-4, 10)  # 纵轴范围，调大上限以留出上方距离
plt.tick_params(axis='both', which='both', length=0)  # 去掉刻度线
plt.legend(loc='upper right', fontsize=10)  # 图例放在右上方
# plt.grid(axis='y', linestyle='--', alpha=0.7)  # 移除背景网格线
plt.tight_layout()  # 自动调整布局，防止标签重叠

# ---------------------- 5. 保存或显示图表 ----------------------
plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
plt.show()