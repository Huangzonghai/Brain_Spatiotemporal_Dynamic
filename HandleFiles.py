import numpy as np
import pandas as pd
import os
import scipy
import pickle

def band_pass_filter_signal(signal, lowcut, highcut, sampling_rate, order=4):
    fft_signal = scipy.fft.fft(signal)
    frequencies = scipy.fft.fftfreq(len(signal), d=1 / sampling_rate)
    cutoff_idx = np.abs(frequencies) > highcut
    fft_signal[cutoff_idx] = 0
    cutoff_idx = np.abs(frequencies) < lowcut
    fft_signal[cutoff_idx] = 0
    filtered_signal = scipy.fft.ifft(fft_signal)
    signal = filtered_signal.real
    return signal

def high_gain_observer(A_do, B_do, C_do, D_do, y, p_k_prev):
    # 计算当前时刻观测器的状态向量p和系统状态的估计值x_hat
    p_k = A_do @ p_k_prev + B_do @ y
    x_hat_k = C_do @ p_k_prev + D_do @ y
    return p_k, x_hat_k

ControlFileList = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv", "8.csv", "9.csv", "10.csv", "11.csv",
                   "12.csv", "13.csv", "14.csv", "15.csv"]
StrokeFileList = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv", "8.csv", "9.csv", "10.csv", "11.csv",
                   "12.csv", "13.csv", "14.csv", "15.csv"]
fileList = []
total_data = []
for p in StrokeFileList:
    fileName = os.path.join(r"D:\Unveiling Cognitive Interference\stroop-control&aphasia\preprocessing data\aphasia", p)
    fileList.append(fileName)
for c in ControlFileList:
    fileName = os.path.join(r"D:\Unveiling Cognitive Interference\stroop-control&aphasia\preprocessing data\control", c)
    fileList.append(fileName)

for f in fileList:
    data = pd.read_csv(f, header=None)
    total_data.append(data)

HbRs = []
HbTs = []
for i in range(len(total_data)):
    hbr = []
    hbt = []
    for j in range(84):
        if j%3==1:
            hbr.append(np.asarray(total_data[i][j]))
        if j%3==2:
            hbt.append(np.asarray(total_data[i][j]))
    HbRs.append(hbr)
    HbTs.append(hbt)

markers = pd.read_csv("marks.csv", header=None).values.T

total_data = []
for i in range(30):
    ones_data = []
    twos_data = []
    threes_data = []
    for j in range(len(HbRs[i])):
        node_data = []
        r = band_pass_filter_signal(HbRs[i][j], 0, 0.1, 10)

        t = HbRs[i][j]
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        B = np.array([[1],[0], [0]])
        C = np.array([[1, 0, 0]])
        h1 = 100
        h2 = 1000
        epsilon = 10
        Ts = 1/250.
        I = np.eye(A.shape[0])
        H0 = np.array([h1, h2]).reshape(-1, 1)
        A0 = A - H0 @ C
        alpha = Ts/epsilon
        D = np.diag([1, epsilon])
        A_do = (I + (alpha / 2) * A0) @ np.linalg.inv(I - (alpha / 2) * A0)
        B_do = alpha * np.linalg.inv(I - (alpha / 2) * A0) @ H0
        C_do = np.linalg.inv(D) @ np.linalg.inv(I - (alpha / 2) * A0)
        D_do = (alpha / 2) * C_do @ H0
        x_0 = np.array([[0], [0]])
        p_0 = np.linalg.inv(C_do)@(x_0-D_do*r[0])
        estimated_states = []
        p_k = p_0
        for y in r:
           y = np.array([[y]])
           p_k, x_hat_k = high_gain_observer(A_do, B_do, C_do,D_do,y ,p_k)
           estimated_states.append(x_hat_k.flatten())

        estimated_states = np.array(estimated_states)

        marker = markers[i]
        num = 0
        mm = []
        otmp = []
        for t in range(len(marker)):
            if marker[t] !=0 and np.isnan(marker[t])!=True:
                mm.append(t)    

        for p in range(0, len(mm)):
            tmp = estimated_states[mm[p]:mm[p] + 120]
            otmp.append(tmp)
        for o in otmp:
            node_data.append(o)
        ones_data.append(node_data)
    if (i) // 15 == 0:
        with open('a' + str(i % 15 + 1) + '_all_total_hbr.pkl', 'wb') as f:
            pickle.dump(ones_data, f)
    else:
        with open('h' + str(i % 15 + 1) + '_all_total_hbr.pkl', 'wb') as f:
            pickle.dump(ones_data, f)
    print(mm[-1]+120, len(estimated_states))
    if mm[-1]+120>=len(estimated_states):
        print(mm, len(estimated_states), len(marker))
        print(j)