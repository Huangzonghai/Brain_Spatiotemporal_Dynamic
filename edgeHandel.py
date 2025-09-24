import numpy as np
import pandas as pd
data = pd.read_csv(r'./datas/pearson_allchanel_hbt_average.csv')
stroke_conn = []
hc_conn = []
for i in range(len(data)):
    dd_stroke = data.iloc[i][:15]
    dd_hc = data.iloc[i][15:]
    stroke_conn.append(np.average(dd_stroke))
    hc_conn.append(np.average(dd_hc))
stroke_mat = np.zeros([53,53])
hc_mat = np.zeros([53,53])
nums = 0
for i in range(53):
    for j in range(53):
        if i == j:
            stroke_mat[i][j] = 1.
            hc_mat[i][j] = 1.
        elif i<j:
            tt = (52+52-i)*i/2+j-i
            stroke_mat[i][j] = stroke_conn[int(tt)]
            hc_mat[i][j] = hc_conn[int(tt)]
        else:
            stroke_mat[i][j] = stroke_mat[j][i]
            hc_mat[i][j] = hc_mat[j][i]
# 保存为 .edge 文件
np.savetxt('stroke.edge', stroke_mat, fmt='%.5f', delimiter=' ')