import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt

path = './A榜测试集数据/csv文件/'
files = os.listdir(path)

label = pd.read_csv('./训练集数据/文件标签汇总数据.csv')

for num in range(len(files)):
    data = pd.read_csv(path + files[num])
    # data = np.array(data)

    plt.gray()
    plt.axis('off')
    plt.scatter(data['X'], data['Y'], c=data['Value'], marker='s')
    plt.savefig('./images/' + files[num][0:-4] + '.jpg', dpi=1000)
    plt.close()

temp = []
for name in label['fileName']:
    name = name[0:-4] + '.jpg'
    temp.append(name)
label = zip(temp, label['defectType'])
label = dict(label)
with open('labels_trainval.yaml', 'w') as f:
    data = yaml.dump(label, f)
