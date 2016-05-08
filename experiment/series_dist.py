# -*- coding: utf-8 -*-

# Module
from similarity import *
from series_transfer import *

# 输入两条Count序列以及beta值， 返回在每个beta上的距离
def cal_dist_vec(beta, list1, list2):
    min_len = min(len(list1), len(list2))

    #距离向量的长度
    interval_num = min_len // beta

    #存放距离向量的列表
    dist_vec = []

    for i in range(interval_num):
        begin = i * beta
        end = begin + beta

        interval_list1 = normalization(list1[begin:end])
        interval_list2 = normalization(list2[begin:end])

        dist_vec.append(euclid_distance(interval_list1, interval_list2))

    return dist_vec

def main():
    list1 = [1,2]
    list2 = [2,3,4]
    print(cal_dist_vec(2, list1, list2))

if __name__ == '__main__':
    main()
