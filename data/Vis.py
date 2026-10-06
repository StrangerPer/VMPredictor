import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def hotpoint(dense_mat, path):
    """
    dense_mat--->torch类型, [T,N]
    """
    dense_mat = dense_mat.numpy().T   # N*T
    bias = 4  
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(12, 2.5))
    sns.heatmap(dense_mat[0:170, bias * 288: (bias + 7) * 288], cmap='jet',
                cbar_kws={'label': 'Traffic speed'}, vmin=0, vmax=1)  # 原始位置是85
    plt.xticks(np.arange(0, 288 * 7 + 1, 288), np.arange(0, 288 * 7 + 1, 288), rotation=0)
    # plt.yticks(np.arange(-1, 90.1, 50))
    plt.xlabel('Time')
    plt.ylabel('Loop detector')
    plt.show()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)

