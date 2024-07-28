import warnings
warnings.filterwarnings("ignore")
from feature_normalized import virtual_samples_predictions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import random
import geatpy as ea

# 输出显示不限制长度
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)



if __name__ == "__main__":
    elements = ["C", "Al", "Si", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Y", "Zr", "Nb", "Mo", "Sn", "Pd",
                "Nd", "Hf", "Ta", "W", "Re"]
    # 注意这里有一个
    elements_range=[[0.051, 0.128], [0.017, 0.333], [0.024, 0.184], [0.2, 0.25], [0.033, 0.333], [0.016, 0.332], [0.025, 0.312],
     [0.057, 0.328], [0.1, 0.347], [0.048, 0.333], [0.111, 0.333], [0.02, 0.328], [0.2, 0.25], [0.143, 0.333],
     [0.029, 0.333], [0.01, 0.286], [0.017, 0.036], [0.172, 0.192], [0.167, 0.1671], [0.091, 0.333],
     [0.051, 0.25], [0.048, 0.286], [0.106, 0.143]]
    HV_count = [4, 336, 7, 2, 107, 58, 369, 45, 404, 328, 382, 209, 2, 34, 49, 78, 2, 0, 1, 28, 27, 13, 5]
    EL_count = [4, 85, 6, 2, 85, 44, 107, 26, 122, 91, 115, 46, 2, 30, 52, 47, 4, 4, 0, 25, 20, 14, 5]
    # 计算各个元素出现的概率
    all_elements_prob = [0.5 * x / sum(HV_count) + 0.5 * y / sum(EL_count) for x, y in zip(HV_count, EL_count)]
    elements_range_length = [elements_range[i][1] - elements_range[i][0] for i in range(len(elements))]


    def element_choose(prob, ele_range_length):
        """
        把每个元素的频率都增大，但是元素之间频率的比值保持不变
        :param prob:
        :param ele_range_length:
        :return:
        """
        negative_range_length = [(1.0/(6.0*prob[i])-1)*ele_range_length[i] for i in range(len(prob))]
        negative_range_start = [round(elements_range[i][0] - negative_range_length[i], 4) for i in range(len(prob))]
        return negative_range_start

    def element_content(var, col_index):
        """
        位于负区间的值归于0，否则等于本身，col_index代表第几列
        :return:
        """
        true_content=np.zeros_like(var)
        for i in range(len(var)):
            #if x[i]>0:
            # TODO 将小于范围下界的值归0
            if var[i]>= elements_range[col_index][0]:
                true_content[i]=var[i]
        return true_content

    

    lb_value=element_choose(all_elements_prob,elements_range_length)
    

    def count(x, k):
        # x代表某一列变量，k代表是整个变量的第k列
        count = np.ones_like(x)
        for i in range(len(x)):
            if x[i]<elements_range[k][0]:
                count[i] = 0
        return count


    def samples_normalized(Vars):
        count_matrix = np.ones_like(Vars)
        for t in range(Vars.shape[1]):
            count_matrix[:, [t]] = count(Vars[:, [t]], t)
        Vars = Vars * count_matrix
        # TODO 找到全是0的行的索引,然后把这行赋值为几个元素使其不为空，否则预测的时候会出错
        all_zero_rows = np.all(count_matrix == 0, axis=1)
        index_all_zero=[i for i in range(Vars.shape[0]) if all_zero_rows[i] == True]
        index_not_all_zero=[i for i in range(Vars.shape[0]) if all_zero_rows[i] == False]

        # print(count_matrix)
        # print(Vars)
        samples = [[] for _ in range(Vars.shape[0])]
        for j in range(Vars.shape[0]):
            if j in index_all_zero:
                Vars[j,:]=Vars[random.choice(index_not_all_zero),:]
            for k in range(Vars.shape[1]):
                if Vars[j, k] > 0:
                    samples[j].append(elements[k] + str(round(Vars[j, k] / sum(Vars[j, :]), 2)))
            samples[j] = ''.join(samples[j])

        return samples




    class MyProblem(ea.Problem):  # 继承Problem父类

        def __init__(self, M=2):
            name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
            Dim = len(elements)  # 初始化Dim（决策变量维数）
            maxormins = [-1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
            varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
            #lb = [i[0] for i in elements_range]  # 决策变量下界
            lb=lb_value
            ub = [i[1] for i in elements_range]  # 决策变量上界
            lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
            ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
            # 调用父类构造方法完成实例化
            ea.Problem.__init__(self,
                                name,
                                M,
                                maxormins,
                                Dim,
                                varTypes,
                                lb,
                                ub,
                                lbin,
                                ubin)

        def evalVars(self, Vars):  # 目标函数
            x1 = Vars[:, [0]]
            x2 = Vars[:, [1]]
            x3 = Vars[:, [2]]
            x4 = Vars[:, [3]]
            x5 = Vars[:, [4]]
            x6 = Vars[:, [5]]
            x7 = Vars[:, [6]]
            x8 = Vars[:, [7]]
            x9 = Vars[:, [8]]
            x10 = Vars[:, [9]]
            x11 = Vars[:, [10]]
            x12 = Vars[:, [11]]
            x13 = Vars[:, [12]]
            x14 = Vars[:, [13]]
            x15 = Vars[:, [14]]
            x16 = Vars[:, [15]]
            x17 = Vars[:, [16]]
            x18 = Vars[:, [17]]
            x19 = Vars[:, [18]]
            x20 = Vars[:, [19]]
            x21 = Vars[:, [20]]
            x22 = Vars[:, [21]]
            x23 = Vars[:, [22]]

           
            CV = np.hstack(
                [count(x1, 0) + count(x2, 1) + count(x3, 2) + count(x4, 3) + count(x5, 4) + count(x6, 5) + count(x7, 6) +
                 count(x8, 7) + count(x9, 8) + count(x10, 9) + count(x11,10) + count(x12, 11) + count(x13,12) +
                 count(x14, 13) + count(x15, 14) + count(x16, 15) + count(x17,16) + count(x18, 17) + count(x19, 18) +
                 count(x20, 19) + count(x21, 20) + count(x22, 21) + count(x23, 22) - 7,
                 4 - (count(x1, 0) + count(x2, 1) + count(x3, 2) + count(x4, 3) + count(x5, 4) + count(x6, 5) + count(x7, 6)
                + count(x8, 7) + count(x9, 8) + count(x10, 9) + count(x11, 10) + count(x12,11) + count(x13, 12) + count(x14, 13) +
                count(x15, 14) + count(x16, 15) + count(x17, 16) + count(x18, 17) + count(x19,18) + count(x20, 19) + count(x21, 20) +
                count(x22, 21) + count(x23, 22)),
                 0.9-(element_content(x1,0)+element_content(x2,1)+element_content(x3,2)+element_content(x4,3)+element_content(x5,4)+element_content(x6,5)+element_content(x7,6)+element_content(x8,7)+element_content(x9,8)+element_content(x10,9)+element_content(x11,10)
                      +element_content(x12,11)+element_content(x13,12)+element_content(x14,13)+element_content(x15,14)+element_content(x16,15)+element_content(x17,16)+element_content(x18,17)+element_content(x19,18)+element_content(x20,19)+element_content(x21,20)+element_content(x22,21)+element_content(x23,22)),
                 element_content(x1,0) + element_content(x2,1) + element_content(x3,2) + element_content(x4,3) + element_content(x5,4) + element_content(x6,5) + element_content(x7,6) + element_content(x8,7) + element_content(x9,8) + element_content(x10,9) + element_content(x11,10)
                      + element_content(x12,11) + element_content(x13,12) + element_content(x14,13) + element_content(x15,14) + element_content(x16,15) + element_content(x17,16) + element_content(x18,17) + element_content(x19,18) + element_content(x20,19) + element_content(x21,20) + element_content(x22,21) +element_content(x23,22)-1.1,
                 ])


            samples=samples_normalized(Vars)
            df_samples = pd.DataFrame()
            df_samples['composition'] = samples

            HV, EL = virtual_samples_predictions(df_samples, composition='composition')

            # np.hstack将参数元组的元素数组按水平方向进行叠加
            f = np.hstack([HV, EL])
            # print(f)
            f = f.reshape(-1, 2, order='F')
            # print(f)
            return f, CV


    # 实例化问题对象
    problem = MyProblem()
    # 构建算法
    algorithm = ea.moea_NSGA2_templet(
    problem,
    # 种群的大小为NIND，原来为200
    # 种群大小设置为500，最大进化代数为10
    # 种群大小设置为200，最大进化代数设置为500
    ea.Population(Encoding='BG', NIND=200),
    MAXGEN=500,  # 最大进化代数
    logTras=100)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.Pm = 0.1  # 修改变异算子的变异概率
    algorithm.recOper.XOVR = 0.7  # 修改交叉算子的交叉概率
    # 求解
    # drawing取1只显示最终结果，2实时绘制目标空间动态图，3实时绘制决策空间动态图
    # 设置为2时图片太多了
    # drawlog是否根据日志绘制迭代变化图像
    # saveFlag尝试的时候设置成FALSE
    res = ea.optimize(algorithm,
              verbose=True,
              drawing=1,
              outputMsg=True,
              drawLog=True,
              saveFlag=True)
    # 结果是一个字典，其中解对应的键是"Vars"
    print(res)

    #保存结果到excel
    print(samples_normalized(res["Vars"]))

    df_results = pd.DataFrame(samples_normalized(res["Vars"]),columns=['formula'])
    df_results["HV"]=res["ObjV"][:,0]
    df_results["EL"]=res["ObjV"][:,1]
    df_results.to_excel("./strategies/strategy2/strategy_compositon.xlsx")