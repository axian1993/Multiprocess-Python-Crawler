# -*- coding: utf-8 -*-

#requirements
from sklearn import cross_validation
from sklearn import svm
import numpy as np
from numpy import *
from numpy import linalg as la
import matplotlib.pyplot as plt

#Module
from similarity import *
from series_transfer import normalization
from series_transfer import Stamps
from user_filter import bhv_cnt_filter

def validation(sim_metric, alpha = 86400, betas = [7,14], bhv_cnt = 50, l_norm = 20, cv = 10 ):

    user_dict = bhv_cnt_filter(bhv_cnt)

    sim = zeros((len(user_dict)**2,len(betas)))
    sim_index = 0
    target = np.array([])

    if sim_metric == "lcss":
        with open('data/zhihu/users_test.json') as z_input, open('data/weibo/users_test.json') as w_input:
            for z_line in z_input:

                z_line = eval(z_line)
                if z_line['index'] not in user_dict:
                    continue

                stamp = Stamps(z_line['activity'])
                z_list = normalization(stamp.contiIntervalCnt(begin = 1458576000, interval = alpha))

                for w_line in w_input:

                    w_line = eval(w_line)

                    if w_line['index'] not in user_dict:
                        continue

                    stamp = Stamps(w_line['activity'])
                    w_list = normalization(stamp.contiIntervalCnt(begin = 1458576000, interval = alpha))

                    min_len = min(len(z_list), len(w_list))
                    plt.plot(range(min_len),z_list)

                    sim_vec = []

                    for beta in betas:
                        interval_num = min(len(z_list)//beta, len(w_list)//beta)
                        in_sim_vec = []

                        for i in range(interval_num):
                            begin = i * beta
                            end = begin + beta
                            #interval_sim = LCSubstring(z_list[begin:end], w_list[begin:end], delta = 0.1)/beta
                            #interval_sim = LCSubsequence(z_list[begin:end], w_list[begin:end], delta = 0.5)/beta
                            interval_sim = 1/edit_distance(z_list[begin:end], w_list[begin:end],0.1)
                            in_sim_vec.append(interval_sim)

                        sim_vec.append(la.norm(in_sim_vec, l_norm))

                    sim[sim_index] = sim_vec
                    sim_index += 1

                    if z_line['index'] == w_line['index']:
                        target = np.append(target, 1)
                        print("*****")
                        print(sim_vec)
                        print("*****")
                    else:
                        target = np.append(target, 0)
                        print(sim_vec)

                w_input.seek(0)
                print("************************************************************")

    # clf = svm.SVC(kernel='linear', C=1)
    # scores = cross_validation.cross_val_score(clf, sim, target, cv = cv)
    # return scores

def filter_test():
    test_user = range(10)
    with open('data/zhihu/users.json', 'r') as input, open('data/zhihu/users_test.json', 'w') as output:
        for line in input:
            line = eval(line)
            if line['index'] in test_user:
                output.write(str(line) + '\n')

def main():
    validation('lcss')
    #filter_test()

if __name__ == "__main__":
    main()
