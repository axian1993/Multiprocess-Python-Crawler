# -*- coding: utf-8 -*-

#requirement
import matplotlib.pyplot as plt
import numpy as np

# Module
from series_transfer import Stamps

def behaviors_cnt():
    zhihu= []
    weibo = []

    with open("data/zhihu/users.json") as z_input, open("data/weibo/users.json") as w_input:
        for line in z_input:
            line = eval(line)
            zhihu.append(len(line['activity']))

        for line in w_input:
            line = eval(line)
            weibo.append(len(line['activity']))

        zhihu_cnt = [0 for n in range(6)]
        weibo_cnt = [0 for n in range(6)]

        for i in zhihu:
            if i == 0:
                zhihu_cnt[0] += 1
            elif i < 50:
                zhihu_cnt[1] += 1
            elif i < 100:
                zhihu_cnt[2] += 1
            elif i < 300:
                zhihu_cnt[3] += 1
            elif i < 1000:
                zhihu_cnt[4] += 1
            else:
                zhihu_cnt[5] += 1

        for i in weibo:
            if i == 0:
                weibo_cnt[0] += 1
            elif i < 50:
                weibo_cnt[1] += 1
            elif i < 100:
                weibo_cnt[2] += 1
            elif i < 300:
                weibo_cnt[3] += 1
            elif i < 1000:
                weibo_cnt[4] += 1
            else:
                weibo_cnt[5] += 1

        x = np.linspace(0, 5, 6)
        y = np.array(zhihu_cnt)

        plt.plot(x, y, color = 'blue', label = 'zhihu')

        x = np.linspace(0, 5, 6)
        y = np.array(weibo_cnt)

        plt.plot(x, y, color = 'red', label = 'weibo')

        plt.xticks(x, ['0', '[1,50)', '[50,100)', '[100,300)', '[300,1000)', '[1000,)'])

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data',0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data',0))

        plt.legend(loc='upper left')

        plt.show()

# 绘制dist_vec的值的分布
def dist_vec_distribution():
    file_path = 'result/dist_vec_mat/day_series_beta_7.txt'
    with open(file_path, 'r') as vec_mat:
        for i in range(1):
            line = vec_mat.readline()
            line = np.array(eval(line))
            for j in range(100):
                plt.clf()
                num_bins = 30
                x = np.array(line[j])
                n, num_bin, patches = plt.hist(x, num_bins, normed = True, color='blue', alpha=0.5)
                print(n)
                title = '{0}_{1}'.format(i, j)
                plt.title(title)
                plt.show()

# 绘制从90天到三个七天的图
def plot_conti_statistic():
    conti_path = 'data/zhihu/cnt/day_series'
    statistic_path = 'data/zhihu/norm_cnt/30_days_weekday'

    with open(conti_path, 'r') as conti, open(statistic_path, 'r') as sta:
        line = eval(conti.readline())
        Y1 = [line['count'][i] for i in range(90)]

        line = eval(sta.readline())
        Y2 = [line['series'][i] for i in range(21)]

    X1 = np.linspace(1, 90, 90)
    plt.plot(X1, Y1)

    # for i in range(3):
    #     X2 = range(i*7 + 1, i*7 + 8)
    #     plt.plot(X2, Y2[i*7: i*7 + 7])
    #
    ax = plt.gca()
    ax.set_xticks([1,30,60,90])
    plt.xlim(1, 90)
    plt.show()

# 绘制每个用户最早的行为记录时间分布
def plot_earliest_distribution():
    platforms = {'zhihu', 'weibo'}

    earliest = np.zeros((1356), dtype = int)

    maxday = 0
    minstamp = 2000000000

    for platform in platforms:
        path = 'data/{0}/users.json'.format(platform)
        with open(path, 'r') as users:
            for user in users:
                user = eval(user)
                index = user['index']
                stamps = Stamps(user['activity'])
                if len(stamps.tsl) != 0:
                    earliest_stamp = stamps.tsl[len(stamps.tsl) - 1]
                    days = (1458576000 - earliest_stamp) // 86400
                    if earliest_stamp < minstamp:
                        minstamp = earliest_stamp
                    if days > maxday:
                        maxday = days
                else:
                    days = 0
                if platform == 'zhihu':
                    earliest[index] = days
                else:
                    earliest[index] = min(earliest[index], days)

    print(minstamp)
    print(maxday)
    num_bins = earliest.max() - earliest.min() + 1
    plt.hist(earliest, num_bins, color = 'blue', alpha = 0.1, normed=True, cumulative=True)
    plt.show()


def main():
    plot_earliest_distribution()

if __name__ == '__main__':
    main()
