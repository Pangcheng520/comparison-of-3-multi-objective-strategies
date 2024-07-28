import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score as CVS
import numpy as np
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
### 模型选择
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import copy
import lightgbm as lgb

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import random
from compute_distance import *
import joblib

# 输出显示不限制长度
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)


def main():

    best_features_HV = ['MagpieData avg_dev CovalentRadius', 'MagpieData mean Electronegativity',
                        'MagpieData avg_dev Electronegativity', 'MagpieData mean NpValence',
                        'MagpieData mean NUnfilled', 'MagpieData avg_dev SpaceGroupNumber',
                        'Mean cohesive energy', 'Shear modulus strength model']
    best_features_elongation = ['MagpieData range MendeleevNumber', 'MagpieData avg_dev MendeleevNumber',
                                'MagpieData avg_dev MeltingT', 'MagpieData mean NValence',
                                'MagpieData mean NsUnfilled', 'MagpieData maximum NUnfilled',
                                'MagpieData maximum GSvolume_pa', 'MagpieData avg_dev GSvolume_pa',
                                'MagpieData minimum SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber',
                                'Lambda entropy', 'Electronegativity local mismatch']
    # 加载两个标准化过程
    std_HV = joblib.load('./strategies/model/std_HV.joblib')
    std_EL = joblib.load('./strategies/model/std_EL.joblib')

    # 建立两个目标函数
    # 硬度模型
    lgbm_HV = joblib.load('./strategies/model/lgbm_feature20_HV.joblib')
    # 压缩应变模型
    svr_EL = joblib.load('./strategies/model/svr_feature20_EL.joblib')

    # ========================虚拟样本筛选==========================================
    #100000个虚拟样本进行筛选
    df_virtual = pd.read_excel("./strategies/Virture_samples_Feature_100000.xlsx")
    
    df2 = df_virtual.copy()

    df2.index = df_virtual["formula"].values
    df2 = df2.drop(["formula"], axis=1)
    print(df2.head())
    Virtual_X_feature_HV = df2[best_features_HV]
    Virtual_X_feature_EL = df2[best_features_elongation]
    Virtual_X_std_feature_HV = std_HV.transform(Virtual_X_feature_HV)
    Virtual_X_std_feature_HV = pd.DataFrame(Virtual_X_std_feature_HV, columns=best_features_HV,
                                            index=df_virtual['formula'].values)
    print(Virtual_X_feature_HV.head())
    Virtual_X_std_feature_EL = std_EL.transform(Virtual_X_feature_EL)
    Virtual_X_std_feature_EL = pd.DataFrame(Virtual_X_std_feature_EL, columns=best_features_elongation,
                                            index=df_virtual['formula'].values)
    #print(Virtual_X_feature_EL.head())
    Virtual_X_std = pd.concat([Virtual_X_std_feature_HV, Virtual_X_std_feature_EL], axis=1)
    num_samples = 3
    filtered_samples_index, filtered_samples_distance = compute_distance(Virtual_X_std, pareto_solvers=Pareto_solves,
                                                                         num_samples=num_samples)
    filtered_samples_index = [j for i in filtered_samples_index for j in i]
    #print(filtered_samples_distance)
    #print(filtered_samples_index)
    df_solvers = pd.DataFrame(filtered_samples_index, columns=["formula"])
    # 按列展开
    df_solvers["distance"] = filtered_samples_distance.reshape((-1, 1), order='F')
    df_solvers["HV"] = lgbm_HV.predict(Virtual_X_std.loc[filtered_samples_index])
    df_solvers["EL"] = svr_EL.predict(Virtual_X_std.loc[filtered_samples_index])
    df_solvers["obj_hv"] = [i for i in Obj.iloc[:, 0] for j in range(num_samples)]
    df_solvers["obj_el"] = [i for i in Obj.iloc[:, 1] for j in range(num_samples)]
    df_solvers.to_excel("./strategies/strategy1/filterd_virtual_samples.xlsx", index=True)


if __name__ == "__main__":
    main()
