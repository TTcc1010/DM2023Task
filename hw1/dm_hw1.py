import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import statistics as st
from sklearn.impute import KNNImputer

folder_path = 'C:\\'  # 数据集文件路径

def reader():
    # DataSet1
    usecols1 = ['YearStart', 'Topic']
    dtype1 = {'YearStart':int}
    data1 = pd.read_csv(folder_path + "Alzheimer Disease and Healthy Aging Data In US.csv", usecols=usecols1, dtype=dtype1,low_memory=False)
    # DataSet2
    usecols2 = ['IMDb-rating', 'downloads', 'run_time', 'views', 'language']
    dtype2 = {'IMDb-rating':float, 'run_time':str}
    data2 = pd.read_csv(folder_path + "movies_dataset.csv",usecols=usecols2, dtype=dtype2)
    # run_time 格式统一
    run_time_min = np.array([])
    for time in data2['run_time']:
        if time != time:
            run_time_min = np.append(run_time_min ,np.NaN)
            continue
        pattern = re.compile(r'\d+')
        times = pattern.findall(str(time))
        if len(times) == 2:
            mins = int(times[0])* 60 + int(times[1]) 
        elif len(times) == 1:
            mins = int(times[0])
        run_time_min = np.append(run_time_min,mins)
    data2['run_time'] = run_time_min
    # downloads & views 转为数值
    data2['downloads'] = data2['downloads'].str.replace(',', '')
    data2['downloads'] = pd.to_numeric(data2['downloads'])
    data2['views'] = data2['views'].str.replace(',', '')
    data2['views'] = pd.to_numeric(data2['views'])
    # print(data1)
    # print(data2)

    return data1, data2

def count(data, str):
    print(data[str].value_counts())

def FiveNum_NullCount(data, str):
    # 五数概括及缺失值计数
    nums = data[str]
    null_cnt = nums.isnull().sum()  # 缺失值
    nums = nums.dropna()            # 删除缺失值
    # 缺失值计数
    print("null_nums:", null_cnt)
    # 五数概括
    print("5 numbers summurize: ",np.percentile(nums, (0, 25, 50, 75, 100)))

def visual_hist(data,):
    # 直方图
    plt.hist(data,align='left')
    plt.show()

def visual_box(data):
    # 盒图
    # 去除缺失值
    nums = data.dropna()
    plt.boxplot(nums)
    plt.show()

if __name__ == "__main__":
    # 读取数据
    data1, data2 = reader()
    # 数据摘要
    # 标称属性
    count(data1, 'Topic')
    count(data2, "language")
    # 五数概括及空缺值计数
    FiveNum_NullCount(data1, 'YearStart')
    FiveNum_NullCount(data2, 'run_time')
    # 可视化
    visual_hist(data1['YearStart'])
    visual_box(data2['run_time'])
    # 填补空缺值
    nums = data2['run_time']
    # 1. 将缺失部分剔除
    nums_p1 = nums.dropna() 
    # 2. 用最高频率值来填补缺失值
    mode = st.mode(nums_p1) # 众数
    nums_p2 = nums.fillna(value=mode)
    # 3. 通过属性的相关关系来填补缺失值
    data = data2.loc[:,['IMDb-rating', 'downloads', 'run_time', 'views']]
    corr_data = data.corr()
    print(corr_data)
    # 4. 通过数据对象之间的相似性来填补缺失值
    data3 = data2.loc[:,['IMDb-rating', 'downloads', 'run_time', 'views']]
    knn = KNNImputer(missing_values=np.NaN, n_neighbors=1)
    data3 = knn.fit_transform(data3)
    nums_p4 = data3[2]
    # 对比数据
    plt.boxplot([nums_p1, nums_p2])
    plt.show()