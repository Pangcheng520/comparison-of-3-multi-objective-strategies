import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import copy
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.svm import SVR
import joblib
# 生成特征相关的库
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition.alloy import WenAlloys

# 输出显示不限制长度
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)




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





def feature_normalized_and_pridict(df_to_be_predicted, composition):
    """

    :param df_to_be_predicted:要预测的样本,composition:成分对应的列名
    :return:虚拟样本硬度和延展性的预测值
    """
    #print('\n', '\033[1mStandardardization on Testing set'.center(120))
    # 要预测的虚拟样本进行特征的标准化
    std_HV = joblib.load('./strategies/model/std_HV.joblib')
    std_EL = joblib.load('./strategies/model/std_EL.joblib')
    df_to_be_predicted.index = df_to_be_predicted[composition].values
    df_to_be_predicted = df_to_be_predicted.drop([composition, "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)
    virtual_samples_std_feature_HV= std_HV.transform(df_to_be_predicted[best_features_HV])
    virtual_samples_std_feature_HV = pd.DataFrame(virtual_samples_std_feature_HV, columns=best_features_HV, index=df_to_be_predicted.index)
    virtual_samples_std_feature_EL = std_EL.transform(df_to_be_predicted[best_features_elongation])
    virtual_samples_std_feature_EL = pd.DataFrame(virtual_samples_std_feature_EL, columns=best_features_elongation,
                                                  index=df_to_be_predicted.index)

    # 建立两个目标函数
    lgbm_HV = joblib.load('./strategies/model/lgbm_HV.joblib')
    svr_EL = joblib.load('./strategies/model/svr_EL.joblib')
    HV = np.array(lgbm_HV.predict(virtual_samples_std_feature_HV))
    EL = np.array(svr_EL.predict(virtual_samples_std_feature_EL))
    return HV,EL







def virtual_samples_predictions(df_virtual_samples, composition):
    """
    # 计算虚拟样本最终20个特征的值并将特征标准化
    :param df_virtual_samples: 虚拟样本的数据框,输入的是成分的式子
    :param composition: 成分对应的列名，字符串
    :return:生成的
    """

    df = StrToComposition(reduce=True, target_col_id='composition_obj').featurize_dataframe(df_virtual_samples,
                                                                                            composition,ignore_errors=True)
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              WenAlloys()])
    data = feature_calculators.featurize_dataframe(df, col_id='composition_obj',ignore_errors=True)


    HV, D = feature_normalized_and_pridict(data,composition=composition)
    return HV, D

if __name__ == '__main__':
    import shap
    import os
    print(os.getcwd())  # 获取当前工作目录路径
    dfH = pd.read_excel("./strategies/HV_Feature.xlsx")
    # 如果有重复值，则保留第一个
    dfH.drop_duplicates(keep='first', inplace=True)
    target = 'HV'
    # 特征的列名
    features = [i for i in dfH.columns if i not in [target]]
    Y_HV = dfH[target]
    X_HV = dfH[features]

    # 训练集，测试集划分
    Train_X_HV, Test_X_HV, Train_Y_HV, Test_Y_HV = train_test_split(X_HV, Y_HV, test_size=0.2, random_state=1)
    HV_Train_X_feature_HV = Train_X_HV[best_features_HV]

    dfD = pd.read_excel("./strategies/ElongationFeature.xlsx")
    # 如果有重复值，则保留第一个
    dfD.drop_duplicates(keep='first', inplace=True)
    # 根据分位数准则删除异常值
    Q1 = dfD['Elongation'].quantile(0.25)
    Q3 = dfD['Elongation'].quantile(0.75)
    IQR = Q3 - Q1
    # 保留小于极端大的值
    dfD = dfD[dfD['Elongation'] <= (Q3 + (1.5 * IQR))]
    # 保留大于极端小的值
    dfD = dfD[dfD['Elongation'] >= (Q1 - (1.5 * IQR))]

    # 删除了一些行，重置行索引
    df1 = dfD.reset_index(drop=True)

    target = 'Elongation'
    # 特征的列名
    features = [i for i in dfD.columns if i not in [target]]
    Y_EL = dfD[target]
    X_EL = dfD[features]

    # 训练集，测试集划分
    Train_X_EL, Test_X_EL, Train_Y_EL, Test_Y_EL = train_test_split(X_EL, Y_EL, test_size=0.2, random_state=59)
    EL_Train_X_feature_EL = Train_X_EL[best_features_elongation]
    EL_Test_X_feature_EL = Test_X_EL[best_features_elongation]

    # Feature Scaling (Standardization)
    std_HV = StandardScaler()
    std_EL = StandardScaler()

    # print('\033[1mStandardardization on Training set'.center(120))
    HV_Train_X_std_feature_HV = std_HV.fit_transform(HV_Train_X_feature_HV)
    HV_Train_X_std_feature_HV = pd.DataFrame(HV_Train_X_std_feature_HV, columns=best_features_HV,
                                             index=Train_X_HV.index)

    EL_Train_X_std_feature_EL = std_EL.fit_transform(EL_Train_X_feature_EL)
    EL_Train_X_std_feature_EL = pd.DataFrame(EL_Train_X_std_feature_EL, columns=best_features_elongation,
                                             index=Train_X_EL.index)

    # 建立两个目标函数
    # 保存 StandardScaler 实例
    joblib.dump(std_HV, './strategies/model/std_HV.joblib')
    joblib.dump(std_EL, './strategies/model/std_EL.joblib')

    # 硬度模型
    lgbm_HV = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                                max_depth=12, num_leaves=15, learning_rate=0.1,
                                n_estimators=690, feature_fraction=1, min_data_in_leaf=20,
                                metric='rmse', random_state=100)

    lgbm_HV.fit(HV_Train_X_std_feature_HV, Train_Y_HV)
    joblib.dump(lgbm_HV, './strategies/model/lgbm_HV.joblib')
    # 压缩应变模型
    svr_EL = SVR(kernel='rbf', C=110, gamma=0.07)
    svr_EL.fit(EL_Train_X_std_feature_EL, Train_Y_EL)
    joblib.dump(svr_EL, './strategies/model/svr_EL.joblib')

    #validated_samples=pd.DataFrame(['Co26Cr19Fe24Mo4Nb5V22','Co26Cr14Fe31Nb13V16','Co32Cr21Fe31Nb16','Al15Co33.5Cr11Fe30.5Mo10'],columns=['composition'])
    #print(validated_samples)
    #HV,D=virtual_samples_predictions(validated_samples,composition='composition')
    #print(HV,D)

    """
    # 计算每个特征的贡献度
    # 创建Explainer
    explainer_lgbm = shap.TreeExplainer(lgbm_HV)
    explainer_svr=shap.KernelExplainer(svr_EL.predict, EL_Train_X_std_feature_EL)
    # 以numpy数组的形式输出SHAP值
    lgbm_shap_values = explainer_lgbm.shap_values(HV_Train_X_std_feature_HV)
    svr_shap_values = explainer_svr.shap_values(EL_Train_X_std_feature_EL)
    # 计算最后一列特征的重要性
    #pd.DataFrame(shap_values).iloc[:, -1].apply(lambda x: abs(x)).mean()  # 输出 3.7574333926117998
    HV_shap_df = pd.DataFrame(np.abs(lgbm_shap_values), columns=best_features_HV)  # 这一步是先取绝对值然后再将array转成pf
    HV_shap_impotance_non_sort=HV_shap_df.mean().values
    print(HV_shap_impotance_non_sort)
    HV_shap_importance = HV_shap_df.mean().sort_values(ascending=False).reset_index()  # 最后计算每个特征的shap值的均值
    HV_shap_importance.columns = ['Feature', 'Shap_Importance']
    print(HV_shap_importance)
    EL_shap_df = pd.DataFrame(np.abs(svr_shap_values), columns=best_features_elongation)  # 这一步是先取绝对值然后再将array转成pf
    EL_shap_impotance_non_sort = EL_shap_df.mean().values
    print(EL_shap_impotance_non_sort)
    EL_shap_importance = EL_shap_df.mean().sort_values(ascending=False).reset_index()  # 最后计算每个特征的shap值的均值
    EL_shap_importance.columns = ['Feature', 'Shap_Importance']
    print(EL_shap_importance)
    """