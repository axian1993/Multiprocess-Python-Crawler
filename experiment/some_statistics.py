# -*- coding: utf-8 -*-

#requirement
import matplotlib.pyplot as plt
import numpy as np

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

def show_vec_mat():
    file_path = 'result/dist_vec_mat/day_series_beta_7.txt'
    with open(file_path, 'r') as vec_mat:
        line = vec_mat.readline()
        line = eval(line)
        for i in range(5):
            for j in range(5):
                print(line[i][j])


def main():
    show_vec_mat()

if __name__ == '__main__':
    main()
