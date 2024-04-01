import numpy as np
from scipy.integrate import simps
from itertools import combinations


def compute_features(BCG, H_locs, I_locs, J_locs, K_locs, L_locs):
    fs = 250
    # 定义你的基准点位置
    locs = {
        "H": H_locs,
        "I": I_locs,
        "J": J_locs,
        "K": K_locs,
        "L": L_locs,
    }

    # 计算各个基准点对应的幅值
    amps = {name: BCG[loc] for name, loc in locs.items()}

    features = {}

    # Time Interval: 基准点位置两两之间的差值

    features["Time Interval"] = {f'{name1}-{name2}': np.abs(loc1 - loc2) / fs for (name1, loc1), (name2, loc2) in
                                 combinations(locs.items(), 2)}

    # Time Ratio: 每一对差值与H-L的差值的占比
    epsilon = 1e-8
    features["Time Ratio"] = {name: np.abs(diff) / (np.abs(features["Time Interval"]["H-L"]) + epsilon) for name, diff in
                              features["Time Interval"].items()}
    del features["Time Ratio"]['H-L']

    # Extremum: 基准点位置对应的信号幅值
    features["Extremum"] = amps

    # Displacement: 两两基准点位置对应的信号幅值的差
    features["Displacement"] = {f'{name1}-{name2}': np.abs(amp1 - amp2) for (name1, amp1), (name2, amp2) in
                                combinations(amps.items(), 2)}

    # AUC: 任意两个基准点之间的BCG信号与X轴之间的面积
    features["AUC"] = {}
    for (name1, loc1), (name2, loc2) in combinations(locs.items(), 2):
        area_list = []
        for i in range(min(len(loc1), len(loc2))):
            start, end = sorted([max(0, min(len(BCG) - 1, loc1[i])), max(0, min(len(BCG) - 1, loc2[i]))])
            if end - start <= 1:  # ensure that there's at least one point between start and end
                area_list.append(0)
                continue
            lower_bound = min(amps["K"][i], amps["I"][i])
            area = simps(np.abs(BCG[start:end] - lower_bound))
            area_list.append(area)
        features["AUC"][f'{name1}-{name2}'] = np.array(area_list)

    return features
