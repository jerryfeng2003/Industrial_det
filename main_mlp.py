import numpy as np
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt

path = './data/A榜测试集数据/csv文件/'
files = os.listdir(path)
label = pd.read_csv('./data/训练集数据/文件标签汇总数据.csv')

x = pd.read_csv(path + files[80])
length = len(x)
# x = np.array(x)
# a_max = np.max(x[:, 0])
# b_max = np.max(x[:, 1])
# x = np.column_stack([np.sqrt(np.square(x[:, 0] * b_max / a_max) + np.square(x[:, 1])), x[:, 2]])
# print(x.shape)
# plt.scatter(x[:, 0], x[:, 1],marker='.')
# plt.show()