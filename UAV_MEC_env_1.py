import numpy as np
import math as mt
import copy
from scipy.optimize import linprog  # 导入 scipy

N_0 = mt.pow(10, ((-169 / 3) / 10))
a = 9.61
b = 0.16
eta_los = 1
eta_nlos = 20
A = eta_los - eta_nlos
C = 20 * np.log10(
    4 * np.pi * 9 / 3) + eta_nlos
B = 2000
Power = 5 * mt.pow(10, 5)
t_min = 1
t_max = 3


class UAV_MEC(object):
    def __init__(self):
        super(UAV_MEC, self).__init__()
        self.N_slot = 400  # number of time slots in one episode
        self.x_s = 10
        self.y_s = 10  # 网格化地图后每一格的长度
        self.h_s = 1  # 每一段的高度
        self.GTs = 6
        self.l_o_v = 100  # initial vertical location
        self.l_f_v = 50 * self.h_s  # final vertical location
        self.l_o_h = [0, 0]  # initial horizontal location 无人机起始水平位置
        self.eps = 20000  # number of episode
        self.energy_max = 80000

        self.f_u = 1000  # 无人机计算速率
        self.f_g = 100  # 用户计算速率

        # up down left right static
        self.action_space_uav_horizontal = ['n', 's', 'e', 'w', 'h']
        # ascend, descend, slf
        self.action_space_uav_vertical = ['a', 'd', 's']  # 三种可能性
        # offloading, local exection
        self.action_space_task_offloading = np.zeros((self.GTs, 2), dtype=np.int)
        # overall_action_space
        self.n_actions = len(self.action_space_uav_horizontal) * len(self.action_space_uav_vertical) * mt.pow(2,
                                                                                                              self.GTs)
        self.n_actions = int(self.n_actions)
        self.param_range = [1, 3]  # 时间的取值范围
        self.n_features = 3  # horizontal:x, y, vertical trajectory of the UAV

        self.actions = np.zeros((np.int(self.n_actions), 1 + 2 + self.GTs), dtype=np.int)
        index = 0

        # 这段代码初始化了动作的空间
        for h in range(len(self.action_space_uav_horizontal)):  # 水平位置
            for v in range(len(self.action_space_uav_vertical)):  # 垂直位置
                # for s in range(self.GTs):
                LL = self.brgd(
                    self.GTs)  # list all the possible combination of 0-1 offloading options among the GTs 64种情况
                for l in range(len(LL)):  # 任务的卸载
                    o_string = LL[l]
                    of = []
                    for ch in range(len(o_string)):
                        if o_string[ch] == '0':
                            of.append(0)
                        else:
                            of.append(1)
                    self.actions[index, :] = [index, h, v] + of[:]
                    index = index + 1
        self._build_uav_mec()

    def _build_uav_mec(self):
        pass

    # 重新开始的无人机的位置
    def reset(self):
        pass

    # 传输速率
    def link_rate(self, gt):
        h = self.l_o_v + self.h_n * self.h_s
        x = self.l_n[0] * self.x_s + 0.5 * self.x_s
        y = self.l_n[1] * self.y_s + 0.5 * self.y_s
        d = np.sqrt(mt.pow(h, 2) + mt.pow(x - self.w_k[gt, 0], 2) + mt.pow(y - self.w_k[gt, 1], 2))

        if (np.sqrt(mt.pow(x - self.w_k[gt, 0], 2) + mt.pow(y - self.w_k[gt, 1], 2)) > 0):
            ratio = h / np.sqrt(mt.pow(x - self.w_k[gt, 0], 2) + mt.pow(y - self.w_k[gt, 1], 2))
        else:
            ratio = np.Inf

        p_los = 1 + a * mt.pow(np.exp(1), (a * b - b * np.arctan(ratio) * (180 / np.pi)))
        p_los = 1 / p_los
        L_km = 20 * np.log10(d) + A * p_los + C
        r = B * np.log2(1 + Power * mt.pow(10, (-L_km / 10)) / (B * N_0))
        return r

    # 传输速率只是参数不同
    def link_rate_single(self, hh, xx, yy, w_k):
        h = self.l_o_v + hh * self.h_s
        x = xx * self.x_s + 0.5 * self.x_s
        y = yy * self.y_s + 0.5 * self.y_s
        d = np.sqrt(mt.pow(h, 2) + mt.pow(x - w_k[0], 2) + mt.pow(y - w_k[1], 2))
        if (np.sqrt(mt.pow(x - w_k[0], 2) + mt.pow(y - w_k[1], 2)) > 0):
            ratio = h / np.sqrt(mt.pow(x - w_k[0], 2) + mt.pow(y - w_k[1], 2))
        else:
            ratio = np.Inf
        p_los = 1 + a * mt.pow(np.exp(1), (a * b - b * np.arctan(ratio) * (180 / np.pi)))
        p_los = 1 / p_los
        L_km = 20 * np.log10(d) + A * p_los + C
        r = B * np.log2(1 + Power * mt.pow(10, (-L_km / 10)) / (B * N_0))
        return r

    # 用户的平衡性
    def balance_ue(self, ue_uav_asso_list):
        pass

    def step(self, action, action_param):
       
        pass

    def find_action(self, index):
        return self.actions[index, :]

    def brgd(self, n):
        pass

    def flight_energy(self, UAV_trajectory, UAV_flight_time, EP, slot):
        d_o = 0.6  # fuselage equivalent flat plate area;
        rho = 1.225  # air density in kg/m3;
        s = 0.05  # rotor solidity;
        G = 0.503  # Rotor disc area in m2;
        U_tip = 120  # tip seep of the rotor blade(m/s);
        v_o = 4.3  # mean rotor induced velocity in hover;
        omega = 300  # blade angular velocity in radians/second;
        R = 0.4  # rotor radius in meter;
        delta = 0.012  # profile drage coefficient;
        k = 0.1  # incremental correction factor to induced power;
        W = 20  # aircraft weight in newton;
        P0 = (delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3))
        P1 = (1 + k) * (pow(W, (3 / 2)) / np.sqrt(2 * rho * G))
        Energy_uav = np.zeros((EP, self.N_slot), dtype=np.float32)
        P2 = 11.46
        count = 0
        for ep in range(self.eps - EP, self.eps):
            horizontal = UAV_trajectory[ep, :, [0, 1]]
            vertical = UAV_trajectory[ep, :, -1]
            t_n = UAV_flight_time[ep, :]

            for i in range(slot[0, ep]):
                if (i == 0):
                    d = np.sqrt((horizontal[0, i] - self.l_o_h[0]) ** 2 + (
                            horizontal[1, i] - self.l_o_h[1]) ** 2)
                    h = np.abs(vertical[i] - vertical[0])
                else:
                    d = np.sqrt(
                        (horizontal[0, i] - horizontal[
                            0, i - 1]) ** 2 + (
                                horizontal[1, i] - horizontal[
                            1, i - 1]) ** 2)
                    h = np.abs(vertical[i] - vertical[i - 1])

                v_h = d / t_n[i]
                v_v = h / t_n[i]
                Energy_uav[count, i] = t_n[i] * P0 * (1 + 3 * np.power(v_h, 2) / np.power(U_tip, 2)) + t_n[i] * (
                        1 / 2) * d_o * rho * s * G * np.power(v_h, 3) + \
                                       t_n[i] * P1 * np.sqrt(
                    np.sqrt(1 + np.power(v_h, 4) / (4 * np.power(v_o, 4))) - np.power(v_h, 2) / (
                            2 * np.power(v_o, 2))) + P2 * v_v * t_n[i]
            count = count + 1
        return Energy_uav

    # 一个时间间隙中的能耗
    def flight_energy_slot(self, pre_l_n, l_n, pre_h, h, t_n):
        d_o = 0.6  # fuselage equivalent flat plate area;
        rho = 1.225  # air density in kg/m3;
        s = 0.05  # rotor solidity;
        G = 0.503  # Rotor disc area in m2;
        U_tip = 120  # tip seep of the rotor blade(m/s);
        v_o = 4.3  # mean rotor induced velocity in hover;
        omega = 300  # blade angular velocity in radians/second;
        R = 0.4  # rotor radius in meter;
        delta = 0.012  # profile drage coefficient;
        k = 0.1  # incremental correction factor to induced power;
        W = 20  # aircraft weight in newton;
        P0 = (delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3))
        P1 = (1 + k) * (pow(W, (3 / 2)) / np.sqrt(2 * rho * G))
        P2 = 11.46

        x_pre = pre_l_n[0] * self.x_s + 0.5 * self.x_s
        y_pre = pre_l_n[1] * self.y_s + 0.5 * self.y_s
        z_pre = self.l_o_v + pre_h * self.h_s
        x = l_n[0] * self.x_s + 0.5 * self.x_s
        y = l_n[1] * self.y_s + 0.5 * self.y_s
        z = self.l_o_v + h * self.h_s

        d = np.sqrt((x_pre - x) ** 2 + (y_pre - y) ** 2)
        h = np.abs(z_pre - z)  #
        v_h = d / t_n
        v_v = h / t_n
        Energy_uav = t_n * P0 * (1 + 3 * np.power(v_h, 2) / np.power(U_tip, 2)) + t_n * (
                1 / 2) * d_o * rho * s * G * np.power(v_h, 3) + \
                     t_n * P1 * np.sqrt(np.sqrt(1 + np.power(v_h, 4) / (4 * np.power(v_o, 4))) - np.power(v_h, 2) / (
                2 * np.power(v_o, 2))) + P2 * v_v * t_n
        return Energy_uav

    # 无人机的轨迹
    def UAV_FLY(self, UAV_trajectory_tmp, slot):
       pass

    # 吞吐量和数据速率
    def throughput(self, UAV_trajectorys, UAV_flight_time, Task_offloadings, UE_Schedulings, Task_offloading_time, EP,
                   slot):
        pass
