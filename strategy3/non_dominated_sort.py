import numpy as np
import pandas as pd
import copy
from strategies.strategy2.feature_normalized import best_features_HV, best_features_elongation
import joblib


def feature_normalized_and_pridict(df_to_be_predicted, composition):
    """

    :param df_to_be_predicted:要预测的样本,composition:成分对应的列名
    :return:虚拟样本硬度和延展性的预测值
    """
    # print('\n', '\033[1mStandardardization on Testing set'.center(120))
    # 要预测的虚拟样本进行特征的标准化
    std_HV = joblib.load('./strategies/model/std_HV.joblib')
    std_EL = joblib.load('./strategies/model/std_EL.joblib')
    df_to_be_predicted.index = df_to_be_predicted[composition].values
    virtual_samples_std_feature_HV = std_HV.transform(df_to_be_predicted[best_features_HV])
    virtual_samples_std_feature_HV = pd.DataFrame(virtual_samples_std_feature_HV, columns=best_features_HV,
                                                  index=df_to_be_predicted.index)
    virtual_samples_std_feature_EL = std_EL.transform(df_to_be_predicted[best_features_elongation])
    virtual_samples_std_feature_EL = pd.DataFrame(virtual_samples_std_feature_EL, columns=best_features_elongation,
                                                  index=df_to_be_predicted.index)

    # 建立两个目标函数
    lgbm_HV = joblib.load('./strategies/model/lgbm_HV.joblib')
    svr_EL = joblib.load('./strategies/model/svr_EL.joblib')
    HV = np.array(lgbm_HV.predict(virtual_samples_std_feature_HV))
    EL = np.array(svr_EL.predict(virtual_samples_std_feature_EL))
    return HV, EL


def compare(p1, p2):
    # return 0同层 1 p1支配p2  -1 p2支配p1
    D = len(p1)
    p1_dominate_p2 = True  # p1 更小
    p2_dominate_p1 = True
    for i in range(D):
        if p1[i] < p2[i]:
            p1_dominate_p2 = False
        if p1[i] > p2[i]:
            p2_dominate_p1 = False

    if p1_dominate_p2 == p2_dominate_p1:
        return 0
    return 1 if p1_dominate_p2 else -1


def fast_non_dominated_sort(P):
    P_size = len(P)
    n = np.full(shape=P_size, fill_value=0)  # 被支配数
    S = []  # 支配的成员
    f = []  # 0 开始每层包含的成员编号们
    rank = np.full(shape=P_size, fill_value=-1)  # 所处等级

    f_0 = []
    for p in range(P_size):
        n_p = 0
        S_p = []
        for q in range(P_size):
            if p == q:
                continue
            cmp = compare(P[p], P[q])
            if cmp == 1:
                S_p.append(q)
            elif cmp == -1:  # 被支配
                n_p += 1
        S.append(S_p)
        n[p] = n_p
        if n_p == 0:
            rank[p] = 0
            f_0.append(p)
    f.append(f_0)

    i = 0
    while len(f[i]) != 0:  # 可能还有i+1层
        Q = []
        for p in f[i]:  # i层中每个个体
            for q in S[p]:  # 被p支配的个体
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        f.append(Q)
    rank += 1
    return rank, f




if __name__ == '__main__':
    df_virtual = pd.read_excel("./strategies/Virture_samples_Feature_100000.xlsx")
    df_formula = pd.read_excel("./strategies/Virtual_samples_100000.xlsx")
    df_virtual['formula'] = df_formula
    HV, EL = feature_normalized_and_pridict(df_to_be_predicted=df_virtual, composition='formula')
    df_100000_high_throught = pd.DataFrame({'formula': df_formula.iloc[:, 0].values, 'H': HV, 'D': EL})
    df_100000_high_throught.to_excel("./strategies/strategy3/100000_samples_prediction.xlsx", index=False)
    
    final_non_dominated_solvers = pd.DataFrame()
    for i in range(6):
        # 一次读取20000个数据
        if i < 5:
            skiprows = 20000 * i
            df = pd.read_excel("./strategies/strategy3/100000_samples_prediction.xlsx", skiprows=skiprows, nrows=20000)
            df.columns = ["formula", "H", "D"]
        else:
            df = final_non_dominated_solvers
        df_ = copy.copy(df)
        df = df[['H', 'D']]
        P = np.array(df)
        # P = np.array([[2, 2, 5], [2, 1, 3], [1, 2, 3], [3, 1, 4], [2, 1, 4], [1, 3, 5], [1, 3, 3]])
        rank, f = fast_non_dominated_sort(P)
        f.pop()  # 去掉最后的空集
        # rank对应pareto等级，从1开始计数，非支配解的等级为1
        # print(rank)
        # 对应从等级1开始对应的解的索引，如这里[[0, 1, 2, 3], [4]]，第一个列表表示等级为1的索引，第二个表示等级为2的索引
        non_dominated_obj = np.zeros((len(f[0]), 2))
        non_dominated_index = f[0]
        print(non_dominated_index)
        # print(non_dominated_index[0])
        for j in range(len(f[0])):
            non_dominated_obj[j, :] = P[non_dominated_index[j], :]
        # print(non_dominated_obj)
        non_dominated_solvers = pd.DataFrame(df_.iloc[non_dominated_index, 0].values, columns=['formula'])
        non_dominated_solvers['H'] = non_dominated_obj[:,0]
        non_dominated_solvers['D']= non_dominated_obj[:,1]
        # print(non_dominated_solvers)
        # 两个数据框的赋值需要索引一样，不一样的索引对应的值会变成NAN，加.values将右边变成数组，没有索引可直接赋值
        final_non_dominated_solvers = final_non_dominated_solvers.append(non_dominated_solvers)
    non_dominated_solvers.to_excel('./strategies/strategy3/results.xlsx', index=False)