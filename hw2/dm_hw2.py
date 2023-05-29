import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# 1. 数据预处理
def preprocess_data(data):
    # 提取用户浏览记录
    user_records = {}

    for index, row in data.iterrows():
        line_type = row['line_type']
        
        if line_type == 'C':
            case_id = row['id']
            user_records[case_id] = []
        elif line_type == 'V':
            vroot_id = row['id']
            user_records[case_id].append(vroot_id)

    visit_records = list(user_records.values())
    print(visit_records)
    
    # 返回预处理后的数据
    return visit_records

# 2. 数据探索性分析
def explore_data(data):
    # 分析最常被访问的页面
    data_v = data.copy()
    data_v = data_v[data_v['line_type'].isin(['V'])]    # 只保留V行
    
    page_visits = data_v['id'].value_counts()
    print("Top 10 most visited pages:")
    print(page_visits.head(10))

    # 绘制页面访问量分布图

    plt.figure(figsize=(12, 6))
    plt.bar(page_visits.index, page_visits.values)
    plt.xlabel('Page ID')
    plt.ylabel('Visit Count')
    plt.title('Page Visit Count Distribution')
    plt.xticks(rotation=90)
    plt.show()
    pass

# 3. 关联规则挖掘
def mine_association_rules(data, min_support, min_confidence):
    # 转换为multi-hot的编码形式
    te = TransactionEncoder()
    te_arr = te.fit(data).transform(data)
    df = pd.DataFrame(te_arr, columns=te.columns_)
    #print(df)

    # 使用Apriori算法计算频繁项集
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    print(frequent_itemsets)
    # 计算关联规则
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_confidence)
    print(rules)
    # 返回关联规则
    return rules

# 4. 结果评估
def evaluate_results(rules, min_support, min_confidence, min_lift):
    
    # 根据支持度、置信度和提升度筛选强关联规则
    strong_rules = rules[(rules['support'] > min_support) & (rules['confidence'] > min_confidence) & (rules['lift'] > min_lift)]
    
    # 返回强关联规则
    return strong_rules


# 主函数
def main():
    # 加载数据集
    folder_path = "D:\\Code\\Python\\dm\\hw2\\"
    data_attribute = pd.read_csv(folder_path + "datasets\\data_attribute.data", header=None, names=['line_type','id','flg','title','url'])
    data_case = pd.read_csv(folder_path + "\\datasets\\data_case.data", header=None, names=['line_type', 'id', 'flg'])
    #print(data_case)
    # 1. 数据预处理
    data_visit = preprocess_data(data_case)
    
    # 2. 数据探索性分析
    explore_data(data_case)
    
    # 3. 关联规则挖掘
    min_support = 0.1  # 最小支持度
    min_confidence = 0.5  # 最小置信度
    min_lift = 0.1  # 最小提升度
    rules = mine_association_rules(data_visit, min_support, min_confidence)
    
    # 4. 结果评估
    strong_rules = evaluate_results(rules,min_support, min_confidence,min_lift)
    print(strong_rules)

if __name__ == "__main__":
    main()
