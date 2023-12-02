#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhangemily
"""

import numpy as np
from tqdm import tqdm


class BasicFDM(object):
    '''
    基本有限差分类, 包含显示、隐式、Crank-Nicolson三种定价方法
    '''

    def __init__(self, S0, r, q, T, vol, S_max, S_min, M, N, method='Crank-Nicolson'):

        self.S0 = S0
        self.r = r
        self.q = q
        self.T = T
        self.vol = vol
        self.S_max = S_max
        self.S_min = S_min
        self.M = M
        self.N = N
        self.delta_S = (S_max - S_min) / M
        self.delta_T = T / N
        
        self.coef = None

        # 行为时间，列为股价
        self.f_mat = np.zeros((N + 1, M + 1))
        
        self._flag_set_boundary = False
        if method not in ['Explicit', 'Implicit', 'Crank-Nicolson']:
            err_message = 'method must be in Explicit, Implicit, Crank-Nicolson'
            raise ValueError(err_message)
        self.method = method


    def set_boundary(self):
        self._flag_set_boundary = True
        return


    def get_abc(self):
        arr = np.linspace(self.S_min, self.S_max, self.M + 1) / self.delta_S
        # 0到M中，只需要1到M-1
        arr = arr[1: self.M]
        
        rjt = (self.r - self.q) * arr * self.delta_T
        sigma2j2t = (self.vol ** 2) * (arr ** 2) * self.delta_T
        
        if self.method == 'Explicit':
            discount = 1 / (1 + (self.r - self.q) * self.delta_T)
            a = discount * (-0.5 * rjt + 0.5 * sigma2j2t)
            b = discount * (1 - sigma2j2t)
            c = discount * (0.5 * rjt + 0.5 * sigma2j2t)
        
        elif self.method == 'Implicit':# 可能溢出
            a = 0.5 * rjt - 0.5 * sigma2j2t
            b = 1 + (self.r-self.q)*self.delta_T + sigma2j2t
            c = -0.5 * rjt - 0.5 * sigma2j2t
            
        else:
            a = -0.25 * rjt + 0.25 * sigma2j2t
            b = -0.5 * (self.r-self.q) * self.delta_T - 0.5 * sigma2j2t
            c = 0.25 * rjt + 0.25 * sigma2j2t
            
        return a, b, c


    def calc_coef(self):
        if self.method == 'Explicit':
            self.explicit_calc_coef()
        elif self.method == 'Implicit':
            self.implicit_calc_coef()
        else:
            self.crank_nicolson_calc_coef()


    def explicit_calc_coef(self):
        if not self._flag_set_boundary:
            raise ValueError('please set f_mat boundary first')
            
        P = np.zeros((self.M - 1, self.M - 1))
        a, b, c = self.get_abc()
        P[0, 0] = b[0]
        P[0, 1] = c[0]
        for i in range(2, self.M - 1):
            P[i - 1, i - 2] = a[i - 1]
            P[i - 1, i - 1] = b[i - 1]
            P[i - 1, i] = c[i - 1]
        P[self.M - 2, self.M - 3] = a[self.M - 2]
        P[self.M - 2, self.M - 2] = b[self.M - 2]

        tempb1 = (a[0] * self.f_mat[:, 0]).reshape(-1, 1)
        tempb2 = np.zeros((self.N + 1, self.M - 3))
        tempb3 = (c[self.M - 2] * self.f_mat[:, self.M]).reshape(-1, 1)
        
        Q = np.hstack((tempb1, tempb2, tempb3))
        self.coef = (P, Q)
        return


    def implicit_calc_coef(self):
        if not self._flag_set_boundary:
            raise ValueError('please set f_mat boundary first')

        P = np.zeros((self.M - 1, self.M - 1))
        a, b, c = self.get_abc()
        P[0, 0] = b[0]
        P[0, 1] = c[0]
        for i in range(2, self.M - 1):
            P[i - 1, i - 2] = a[i - 1]
            P[i - 1, i - 1] = b[i - 1]
            P[i - 1, i] = c[i - 1]
        P[self.M - 2, self.M - 3] = a[self.M - 2]
        P[self.M - 2, self.M - 2] = b[self.M - 2]
        
        tempb1 = (a[0] * self.f_mat[:, 0]).reshape(-1, 1)
        tempb2 = np.zeros((self.N + 1, self.M - 3))
        tempb3 = (c[self.M - 2] * self.f_mat[:, self.M]).reshape(-1, 1)
        Q = np.hstack((tempb1, tempb2, tempb3))
        Pinv = np.linalg.inv(P)
        self.coef = (Pinv, Q)
        return


    def crank_nicolson_calc_coef(self):
        a, b, c = self.get_abc()
        P1 = np.zeros((self.M - 1, self.M - 1))
        P1[0, 0] = 1 - b[0]
        P1[0, 1] = -c[0]
        for i in range(2, self.M - 1):
            P1[i - 1, i - 2] = -a[i - 1]
            P1[i - 1, i - 1] = 1 - b[i - 1]
            P1[i - 1, i] = -c[i - 1]
        P1[self.M - 2, self.M - 3] = -a[self.M - 2]
        P1[self.M - 2, self.M - 2] = 1 - b[self.M - 2]
        
        P2 = np.zeros((self.M - 1, self.M - 1))
        P2[0, 0] = 1 + b[0]
        P2[0, 1] = c[0]
        for i in range(2, self.M - 1):
            P2[i - 1, i - 2] = a[i - 1]
            P2[i - 1, i - 1] = 1 + b[i - 1]
            P2[i - 1, i] = c[i - 1]
        P2[self.M - 2, self.M - 3] = a[self.M - 2]
        P2[self.M - 2, self.M - 2] = 1 + b[self.M - 2]
        
        tempb1 = (a[0] * self.f_mat[:, 0]).reshape(-1, 1)
        tempb2 = np.zeros((self.N + 1, self.M - 3))
        tempb3 = (c[self.M - 2] * self.f_mat[:, self.M]).reshape(-1, 1)
        Q = np.hstack((tempb1, tempb2, tempb3))
        P1inv = np.linalg.inv(P1)
        self.coef = (P1inv, P2, Q)
        return


    def get_fi_1(self, i):
        fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
        if self.method == 'Explicit':
            P, Q = self.coef
            Qi = Q[i, :].reshape(-1, 1)
            fi_1 = P @ fi + Qi
        
        elif self.method == 'Implicit':
            Pinv, Q = self.coef
            Qi_1 = Q[i-1, :].reshape(-1, 1)
            fi_1 = Pinv @ (fi - Qi_1)
        
        else:
            P1inv, P2, Q = self.coef
            Qi = Q[i, :].reshape(-1, 1)
            Qi_1 = Q[i-1, :].reshape(-1, 1)
            fi_1 = P1inv@(P2 @ fi + Qi + Qi_1)
        fi_1 = fi_1.reshape(-1)
        
        return fi_1


    def calc_f_mat(self):
        if not self._flag_set_boundary:
            raise ValueError('please set f_mat boundary first')

        self.calc_coef()
        for i in tqdm(range(self.N, 0, -1)):
            fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
            fi_1 = self.get_fi_1(i)
            self.f_mat[i - 1, 1:self.M] = fi_1
        return


    def interpolate_f_value(self):
        idx = int((self.S0 - self.S_min) / self.delta_S)
        f_upper = self.f_mat[0, idx + 1]
        f_lower = self.f_mat[0, idx]
        s_upper = self.S_min + self.delta_S * (idx + 1)
        s_lower = self.S_min + self.delta_S * idx
        
        df_ds = (f_upper - f_lower) / (s_upper - s_lower)
        f_value = f_lower + df_ds * (self.S0 - s_lower)
        return f_value


    def get_f_value(self):
        self.set_boundary()
        self.calc_f_mat()
        f_value = self.interpolate_f_value()
        return f_value



    ########################################################
    def get_f_delta(self, S, flag_calc_f_mat=False):
        if not flag_calc_f_mat:
            self.set_boundary()
            # 计算边界价格
            self.calc_f_mat()
        idx_S = int((S - self.S_min) / self.delta_S)
        f2 = self.f_mat[0, idx_S+1]
        f1 = self.f_mat[0, idx_S-1]
        delta = (f2 - f1)/self.delta_S
        return delta
    
    
    def get_f_theta(self, t, flag_calc_f_mat=False):
        if not flag_calc_f_mat:
            self.set_boundary()
            # 计算边界价格
            self.calc_f_mat()
        idx_S = int((self.S0-self.S_min)/self.delta_S)
        idx_T = int(t/self.delta_T)
        f1 = self.f_mat[idx_T, idx_S]
        f2 = self.f_mat[idx_T+1, idx_S]
        theta = (f2 - f1)/self.delta_T
        return theta