# -*- coding: utf-8 -*-

#built-in/std
import os

# requirements
from sklearn import cross_validation
from sklearn import svm
import numpy as np
from numpy import *
from numpy import linalg as la
import matplotlib.pyplot as plt

# Module
from similarity import *
from series_transfer import normalization
from series_transfer import Stamps
from user_filter import bhv_cnt_filter

zhihu_path = 'data/zhihu/users_test.json'
weibo_path = 'data/weibo/users_test.json'

def validation(alpha = 86400, betas = [7], bhv_cnt = 100, l_norm = 1.3, cv = 10 ):

    user_dict = bhv_cnt_filter(bhv_cnt, zhihu_path, weibo_path)

    sim = zeros((len(user_dict)**2,len(betas)))
    sim_index = 0
    target = np.array([])

    with open(zhihu_path, 'r') as z_input, open(weibo_path, 'r') as w_input:
        for z_line in z_input:

            z_line = eval(z_line)
            if z_line['index'] not in user_dict:
                continue

            stamp = Stamps(z_line['activity'])
            z_list = stamp.contiIntervalCnt(begin = 1458576000, interval = alpha)

            for w_line in w_input:

                w_line = eval(w_line)

                if w_line['index'] not in user_dict:
                    continue

                stamp = Stamps(w_line['activity'])
                w_list = stamp.contiIntervalCnt(begin = 1458576000, interval = alpha)

                min_len = min(len(z_list), len(w_list))

                if z_line['index'] == w_line['index']:
                    target = np.append(target, 1)
                    print("*****")
                else:
                    target = np.append(target, 0)

                sim_vec = []

                for beta in betas:
                    interval_num = min(len(z_list)//beta, len(w_list)//beta)
                    print(interval_num)
                    in_sim_vec = []

                    for i in range(interval_num):
                        begin = i * beta
                        end = begin + beta

                        zbe = normalization(z_list[begin:end])
                        wbe = normalization(w_list[begin:end])

                        interval_sim = np.exp(0 - ddtw_wdistance(zbe, wbe, 0))

                        if z_line['index'] == 0 and w_line['index'] != 0:
                            # print(zbe)
                            # print(wbe)
                            print(interval_sim)

                            plt.plot(range(beta),zbe, color = 'blue', label='zhihu')
                            plt.plot(range(beta),wbe, color = 'red', label='weibo')

                            ax = plt.gca()
                            ax.spines['right'].set_color('none')
                            ax.spines['top'].set_color('none')
                            ax.xaxis.set_ticks_position('bottom')
                            ax.spines['bottom'].set_position(('data',0))
                            ax.yaxis.set_ticks_position('left')
                            ax.spines['left'].set_position(('data',0))

                            plt.legend(loc='upper left')

                            plt.show()
                        #interval_sim = LCSubstring(z_list[begin:end], w_list[begin:end], delta = 0.1)/beta
                        #interval_sim = LCSubsequence(z_list[begin:end], w_list[begin:end], delta = 0.5)/beta
                        #interval_sim = 1/edit_distance(z_list[begin:end], w_list[begin:end],0.1)

                        in_sim_vec.append(interval_sim)

                    sim_vec.append(la.norm(in_sim_vec, l_norm))

                sim[sim_index] = sim_vec
                sim_index += 1

                print(sim_vec)


            w_input.seek(0)
            print("************************************************************")


    # clf = svm.SVC(kernel='linear', C=1)
    # scores = cross_validation.cross_val_score(clf, sim, target, cv = cv)
    # return scores

def cal_dist(list1, list2, beta, metric, window = 0, symmetry = False ,smooth = False):

    min_len = min(len(list1), len(list2))

    if metric == "dtw":
        interval_num = min_len // beta
        dist_vec = []

        for i in range(interval_num):
            begin = i * beta
            end = begin + beta

            interval_list1 = normalization(list1[begin:end])
            interval_list2 = normalization(list2[begin:end])

            dist_vec.append(dtw_wdistance(interval_list1, interval_list2, window, symmetry))

    elif metric == "ddtw":
        interval_num = min_len // beta
        dist_vec = []

        for i in range(interval_num):
            begin = i * beta
            end = begin + beta

            interval_list1 = normalization(list1[begin:end])
            interval_list2 = normalization(list2[begin:end])

            dist_vec.append(ddtw_wdistance(interval_list1, interval_list2, window, symmetry, smooth))

    return dist_vec

def sim_write():
    alpha_beta = [[3600, [12, 24, 48]], [86400, [7, 14, 30]], [604800, [4, 12, 24]], [2592000, [6, 12]]]
    metric_sym_smo = [['dtw', False, False], ['dtw', True, False], ['ddtw', False, False], ['ddtw', False, True], ['ddtw', True, False], ['ddtw', True, True]]
    bhv_cnts = [100, 300, 1000, 10000]

    with open(zhihu_path, 'r') as zhihu, open(weibo_path, 'r') as weibo:
        user_num = sum(1 for x in zhihu)
        zhihu.seek(0)

        for bhv_cnt in bhv_cnts:
            user_dict = bhv_cnt_filter(bhv_cnt, zhihu_path, weibo_path)
            print("bhv_cnt:%d" %bhv_cnt)

            for alpha, betas in alpha_beta:
                print("alpha:%d" %alpha)
                sim = np.zeros((len(metric_sym_smo), len(betas), user_num, user_num))

                for z_user in zhihu:
                    z_user = eval(z_user)

                    if z_user['index'] not in user_dict:
                        continue

                    for w_user in weibo:

                        w_user = eval(w_user)

                        if w_user['index'] not in user_dict:
                            continue

                        z_list = Stamps(z_user['activity']).contiIntervalCnt(begin = 1458576000, interval = alpha)
                        w_list = Stamps(w_user['activity']).contiIntervalCnt(begin = 1458576000, interval = alpha)

                        for i in range(len(metric_sym_smo)):
                            metric,symmetry,smooth = metric_sym_smo[i]
                            sim_vec = cal_sim(z_list, w_list, betas, metric, symmetry = symmetry, smooth = smooth)
                            for j in range(len(sim_vec)):
                                sim[i][j][z_user['index']][w_user['index']] = sim_vec[j]

                    weibo.seek(0)
                zhihu.seek(0)

                out_path = "sim/alpha_%d_bhv_%d" %(alpha,bhv_cnt)
                with open(out_path, 'w+') as output:
                    output.write(str(sim))

def rank_write():
    alpha_beta = [[3600, [12, 24, 48]], [86400, [7, 14, 30]], [604800, [4, 12, 24]], [2592000, [6, 12]]]
    metric_sym_smo = [['dtw', False, False], ['dtw', True, False], ['ddtw', False, False], ['ddtw', False, True], ['ddtw', True, False], ['ddtw', True, True]]
    bhv_cnts = [100, 300, 1000, 10000]
    alpha_key = ['1hour', '1day', '1week', '1month']

    with open(zhihu_path, 'r') as zhihu:
        user_num = sum(1 for x in zhihu)

    rank_dict = {}

    for bhv_cnt in bhv_cnts:
        for k,alpha,betas in enumerate(alpha_beta):
            sim_path = "sim/bhv_%d_alpha_%d"%(bhv_cnt,alpha)
            with open(sim_path, 'r') as sim:
                sim_mat = eval(sim.readline())

            for i in range(len(metric_sym_smo)):
                for j in range(len(betas)):
                    rank_cnt = rank_distribution(sim_mat[i][j])

                    key = "%d_%s_%d_%s"%(bhv_cnt, alpha_key[k], betas[j])
                    if metric_sym_smo[i][0] == 'dtw':
                        key += '_dwt'
                        if metric_sym_smo[i][1] == False:
                            key += '_nsym'
                        else:
                            key += '_sym'
                    else:
                        key += '_ddwt'
                        if metric_sym_smo[i][1] == False:
                            key += '_nsym'
                        else:
                            key += '_sym'
                        if metric_sym_smo[i][2] == False:
                            key += '_nsmo'
                        else:
                            key += '_smo'

                    rank_dict[key] = rank_cnt

    rank_path = 'result/rank_distribution'
    with open(rank_path, 'w+') as output:
        output.write(str(ran_dict))


def rank_distribution(sim_mat):
    row, col = sim_mat.shape

    row_rank_cnt = [0 for i in range(col)]
    col_rank_cnt = [0 for i in range(row)]

    for i in range(row):
        pos_sim = sim_mat[i][i]
        sorted_sim = sorted(sim_mat[i], reverse = True)
        for j in range(len(sorted_sim)):
            if sorted_sim[j] == pos_sim:
                row_rank_cnt[j] += 1
                break

    for i in range(col):
        pos_sim = sim_mat[i][i]
        sorted_sim = sorted(sim_mat[:][i], reverse = True)
        for j in range(len(sorted_sim)):
            if sorted_sim[j] == pos_sim:
                col_rank_cnt[j] += 1
                break

    return [row_rank_cnt, col_rank_cnt]

def rank_plot():
    with open(zhihu_path, 'r') as zhihu:
        user_num = sum(1 for x in zhihu)

    rank_path = 'result/rank_distribution'

    with open(rank_path, 'r') as rank:
        rank_dict = eval(rank.readline())

    keys = ['100_1hour_12_dtw_nsym']

    for key in keys:

        plt.plot(range(1, user_num+1),rank_dict[key][0], color = 'blue',label='zhihu_'+key)
        plt.plot(range(1, user_num+1),rank_dict[key][1], color = 'red',label='weibo_'+key)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    plt.legend(loc='upper left')

    plt.show()



def filter_test():
    test_user = range(10)
    with open('data/zhihu/users.json', 'r') as input, open('data/zhihu/users_test.json', 'w') as output:
        for line in input:
            line = eval(line)
            if line['index'] in test_user:
                output.write(str(line) + '\n')

def main():
    validation()
    #filter_test()
    #sim_write()
    #rank_write()

if __name__ == "__main__":
    main()
