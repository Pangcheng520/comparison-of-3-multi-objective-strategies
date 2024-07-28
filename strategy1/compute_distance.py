import numpy as np
import pandas as pd

df = pd.read_excel("./strategies/strategy1/NSGA2_feature_solve.xlsx")
Pareto_solves = df.iloc[:, :-2]
Obj = df.iloc[:, [-2, -1]]

HV_shap = [23.86376467, 18.05691411, 14.50906777, 64.20680646, 78.37813164, 14.04499322,
           22.45440617, 83.60254587]
HV_shap_weights = np.array([i / sum(HV_shap) for i in HV_shap])
EL_shap = [2.29105744, 1.49381635, 1.55551178, 1.31423297, 2.25535041, 1.91867468,
           1.96906963, 4.94933092, 1.47353625, 1.41080768, 1.10393329, 6.17124573]
EL_shap_weights = np.array([i / sum(EL_shap) for i in EL_shap])



def compute_distance(virtual_samples, pareto_solvers, num_samples):
    """

    :param virtual_samples: dataframe，索引为化学式
    :param pareto_solvers:
    :param num_samples:每个帕累托非支配解对应的最接近的虚拟样本数量
    :return:
    """
    m = pareto_solvers.shape[0]
    n = len(virtual_samples.iloc[:, 0])
    distance_matrix = np.ones((n, m))
    distance_matrix_rank = np.ones((n, m))
    filtered_samples_distance = np.ones((num_samples, m))
    filtered_samples_index = [list(np.ones(num_samples)) for i in range(m)]

    for i in range(n):
        for j in range(m):
            a_HV = np.array(virtual_samples.iloc[i, :8])
            a_EL = np.array(virtual_samples.iloc[i, 8:])
            b_HV = np.array(pareto_solvers.iloc[j, :8])
            b_EL = np.array(pareto_solvers.iloc[j, 8:])
            distance_matrix[i, j] = np.sqrt(np.sum(np.multiply(HV_shap_weights, np.square(a_HV - b_HV))))+np.sqrt(np.sum(np.multiply(EL_shap_weights, np.square(a_EL - b_EL))))
            
    distance_matrix_df = pd.DataFrame(distance_matrix, index=virtual_samples.index)

    for i in range(m):
        distance_matrix_rank[:, i] = distance_matrix_df.iloc[:, i].rank(method="first")
        for j in range(n):
            if distance_matrix_rank[j, i] in range(1, num_samples + 1):
                rank_ = int(distance_matrix_rank[j, i])
                filtered_samples_distance[rank_ - 1, i] = distance_matrix[j, i]
                filtered_samples_index[i][rank_ - 1] = virtual_samples.index[j]

    return filtered_samples_index, filtered_samples_distance
