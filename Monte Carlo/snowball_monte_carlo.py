import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(0)
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 


class snowball():
    
    def __init__(self, vol, ki, ko, sd, T, coupon, div, IM, q, rf, steps):
        self.vol = vol
        self.ki = ki
        self.ko = ko
        self.sd = sd
        self.T = T
        self.coupon = coupon
        self.div = div
        self.IM = IM
        self.q = q
        self.rf = rf
        self.steps = steps

    def stock_MC(self, sim = 300000):
        dt = 1/self.steps
        price = np.zeros((self.T + 1, sim))
        price[0] = 1
        for i in range(1, self.T + 1):
            z = np.random.standard_normal(sim)
            price[i] = price[i - 1] * np.exp((self.rf - self.q - 0.5 * self.vol ** 2) * dt + self.vol * np.sqrt(dt) * z)
        return price.T


    def snowball_pricing(self, S0 = 1, sim = 10000):
        prices = self.stock_MC(sim)
        
        values = []
        for i in tqdm(range(len(prices))):
            price = prices[i]
            value = None
            days = np.arange(0, price.shape[0], 1)
            up_B2 = (price >= S0 * self.ko)      # 高于敲出价格的日期
            ko_observe = (days % 21 == 0)            # 敲出观察日
            ko_days = days[up_B2 * ko_observe]         # 敲出日
            
            ki_days = days[price < S0 * self.ki]
            
            if len(ko_days) > 0:
                end_day = ko_days[0]
                value = self.coupon * (end_day / self.steps) * np.exp(-self.rf * end_day / self.steps)
            
            elif len(ko_days) == 0 and len(ki_days) > 0:
                end_day = self.T
                if price[-1] < S0:
                    value = price[-1] - 1
        
                else:
                    value = 0
                    
            elif len(ko_days) == 0 and len(ki_days) == 0:
                end_day = self.T
                value = self.div * (end_day / self.steps) * np.exp(-self.rf * end_day / self.steps)
            values.append(value)
            
        return values
    
    def plot_distribution(self, values):
    # 绘制收益分布图
        plt.figure(figsize=(7, 4), dpi=180)
        plt.hist(values, bins=20)
        plt.title('Snowball Price Distribution')
        plt.show()



if __name__ == '__main__':
    
    S0 = 1
    vol = 0.15
    ki = 0.8
    ko = 1.03
    sd = 0.05
    T = 252
    coupon = 0.2
    div = 0.2
    IM = 0.2
    q = -0.032
    rf = 0.02
    sim = 300000
    steps = 252
    
    sb = snowball(vol, ki, ko, sd, T, coupon, div, IM, q, rf, steps)
    values = sb.snowball_pricing(S0, sim = sim)
    price = np.mean(values)
    print()
    print(f"期权价值: {price: 0.6%}")
    
    sb.plot_distribution(values)
    

