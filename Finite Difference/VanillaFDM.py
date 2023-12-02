#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:16:07 2023

@author: zhangemily
"""

import numpy as np
from BasicFDM import BasicFDM
from bs import euro_bs_call, euro_bs_put


class VanillaFDM(BasicFDM):

    def __init__(self,
                 S0 = 1,
                 K = 1,
                 r = 0.03,
                 q = 0,
                 T = 1,
                 vol = 0.13,
                 S_min = 0,
                 S_max = 2,
                 M = 100,
                 N = 252,
                 method = 'Crank-Nicolson',
                 opt_type = 'call'):

        self.K = K
        super().__init__(S0, r, q, T, vol, S_max, S_min, M, N, method)
        if opt_type not in ['call', 'put']:
            raise ValueError('opt_type must be call or put!')
        self.opt_type = opt_type


    def set_boundary(self):
        if self.opt_type == 'call':
            self._call_set_boundary()
        else:
            self._put_set_boundary()
        self._flag_set_boundary = True


    def _call_set_boundary(self):
        
        # 边界条件1： 股票价格为S_min时，期权价格为0
        self.f_mat[:, 0] = 0
        
        # 边界条件2：到期时间T时，期权价格f=max(ST-K, 0)
        for j in range(self.M + 1):
            ST = self.S_min + j * self.delta_S
            self.f_mat[self.N, j] = np.maximum(ST - self.K, 0)
            
        # 边界条件3：S=S_max时，call=S_max-Kexp(-r(T-t))
        for i in range(self.N + 1):
            ti = self.T - i * self.delta_T
            self.f_mat[i, self.M] = self.S_max - self.K * np.exp(-self.r * ti)
        self._init_f_mat_flag = True


    def _put_set_boundary(self):
        # 边界条件1： 初始价格为S_min时，期权价格为f=Kexp(-r(T-t))-Smin
        for i in range(self.N + 1):
            ti = self.T - i * self.delta_T
            self.f_mat[i, 0] = self.K * np.exp(-self.r * ti) - self.S_min
            
        # 边界条件2：到期时间T时，期权价格f=max(K-ST, 0)
        for j in range(self.M + 1):
            ST = self.S_min + j * self.delta_S
            self.f_mat[self.N, j] = np.maximum(self.K - ST, 0)
            
        # 边界条件3：S=S_max时，put=0
        self.f_mat[:, self.M] = 0
        self._init_f_mat_flag = True


if __name__ == '__main__':
    S0 = 1
    K = 1
    r = 0.03
    q = 0
    T = 1
    vol = 0.2
    S_max = 2
    S_min = 0
    M = 1000
    N = 252
    
    fdm_call = VanillaFDM(S0=S0,
                          K=K,
                          r=r,
                          q=q,
                          T=T,
                          vol=vol,
                          S_max=S_max,
                          S_min=S_min,
                          M=M,
                          N=N,
                          method='Crank-Nicolson',
                          opt_type='call')
    vanilla_call = fdm_call.get_f_value()
    print('FDM call: ', vanilla_call)
    print('BS  call: ', euro_bs_call(S0, K, r, T, sigma))
    print()
    
    fdm_put = VanillaFDM(S0=S0,
                         K=K,
                         r=r,
                         T=T,
                         vol=vol,
                         S_max=S_max,
                         S_min=S_min,
                         M=M,
                         N=N,
                         opt_type='put')
    vanilla_put = fdm_put.get_f_value()
    print()
    print('FDM put: ', vanilla_put)
    print('BS  put: ', euro_bs_put(S0, K, r, T, sigma))