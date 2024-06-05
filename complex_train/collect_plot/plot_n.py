# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# Load data from text file using numpy
data = np.loadtxt('output_result_x.txt')
# epoch:  rmse:   mse:    pearson spearman        ci

aspect_ratio = 2.5
width = 8
height = width / aspect_ratio
fig = plt.figure(figsize=(width, height))

# Panel 1
plt.subplot(2, 3, 1)
plt.plot(data[:, 0], data[:, 1], color='blue')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.ylim([0, 1])  # 设置 Y 轴范围为 0 到 1
plt.yticks(np.linspace(0, 1, 5))  # 设置 Y 轴刻度为 5 个数字


# Panel 2
plt.subplot(2, 3, 2)
plt.plot(data[:, 0], data[:, 2], color='red')
plt.xlabel('Epoch')
plt.ylabel('TPR')
plt.ylim([0, 1])  # 设置 Y 轴范围为 0 到 1
plt.yticks(np.linspace(0, 1, 5))  # 设置 Y 轴刻度为 5 个数字

# Panel 3
plt.subplot(2, 3, 3)
plt.plot(data[:, 0], data[:, 3], color='green')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.ylim([0, 1])  # 设置 Y 轴范围为 0 到 1
plt.yticks(np.linspace(0, 1, 5))  # 设置 Y 轴刻度为 5 个数字



# Panel 4
plt.subplot(2, 3, 4)
plt.plot(data[:, 0], data[:, 4], color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])  # 设置 Y 轴范围为 0 到 1
plt.yticks(np.linspace(0, 1, 5))  # 设置 Y 轴刻度为 5 个数字



# Panel 5
plt.subplot(2, 3, 5)
plt.plot(data[:, 0], data[:, 5], color='purple')
plt.xlabel('Epoch')
plt.ylabel('MCC')
plt.ylim([-1, 1])  # 设置 Y 轴范围为 -1 到 1
plt.yticks(np.linspace(-1, 1, 5))  # 设置 Y 轴刻度为 5 个数字


# 调整子图之间的间距
plt.subplots_adjust(wspace=0.5, hspace=1.2, bottom=0.25)

# 显示图形
# plt.show()

plt.savefig('output.png', dpi=300)




