

#built-in/std
import numpy as np
from numpy import *
from scipy import stats
from scipy import io as spio
import matplotlib.pyplot as plt
import time
import multiprocessing
from math import sqrt
from heapq import nlargest
import random

# Module
from similarity import *
from series_transfer import *
from user_filter import bhv_cnt_filter

zhihu_path = 'data/zhihu/users_test.json'
weibo_path = 'data/weibo/users_test.json'

# def validation(alpha = 86400, betas = [7], bhv_cnt = 100, l_norm = 1.3, cv = 10 ):
#
#     user_dict = bhv_cnt_filter(bhv_cnt, zhihu_path, weibo_path)
#
#     sim = zeros((len(user_dict)**2,len(betas)))
#     sim_index = 0
#     target = np.array([])
#
#     with open(zhihu_path, 'r') as z_input, open(weibo_path, 'r') as w_input:
#         for z_line in z_input:
#
#             z_line = eval(z_line)
#             if z_line['index'] not in user_dict:
#                 continue
#
#             stamp = Stamps(z_line['activity'])
#             z_list = stamp.contiIntervalCnt(begin = 1458576000, interval = alpha)
#
#             for w_line in w_input:
#
#                 w_line = eval(w_line)
#
#                 if w_line['index'] not in user_dict:
#                     continue
#
#                 stamp = Stamps(w_line['activity'])
#                 w_list = stamp.contiIntervalCnt(begin = 1458576000, interval = alpha)
#
#                 min_len = min(len(z_list), len(w_list))
#
#                 if z_line['index'] == w_line['index']:
#                     target = np.append(target, 1)
#                     print("*****")
#                 else:
#                     target = np.append(target, 0)
#
#                 sim_vec = []
#
#                 for beta in betas:
#                     interval_num = min(len(z_list)//beta, len(w_list)//beta)
#                     print(interval_num)
#                     in_sim_vec = []
#
#                     for i in range(interval_num):
#                         begin = i * beta
#                         end = begin + beta
#
#                         zbe = normalization(z_list[begin:end])
#                         wbe = normalization(w_list[begin:end])
#
#                         interval_sim = np.exp(0 - ddtw_wdistance(zbe, wbe, 0))
#
#                         if z_line['index'] == 0 and w_line['index'] != 0:
#                             # print(zbe)
#                             # print(wbe)
#                             print(interval_sim)
#
#                             plt.plot(range(beta),zbe, color = 'blue', label='zhihu')
#                             plt.plot(range(beta),wbe, color = 'red', label='weibo')
#
#                             ax = plt.gca()
#                             ax.spines['right'].set_color('none')
#                             ax.spines['top'].set_color('none')
#                             ax.xaxis.set_ticks_position('bottom')
#                             ax.spines['bottom'].set_position(('data',0))
#                             ax.yaxis.set_ticks_position('left')
#                             ax.spines['left'].set_position(('data',0))
#
#                             plt.legend(loc='upper left')
#
#                             plt.show()
#                         #interval_sim = LCSubstring(z_list[begin:end], w_list[begin:end], delta = 0.1)/beta
#                         #interval_sim = LCSubsequence(z_list[begin:end], w_list[begin:end], delta = 0.5)/beta
#                         #interval_sim = 1/edit_distance(z_list[begin:end], w_list[begin:end],0.1)
#
#                         in_sim_vec.append(interval_sim)
#
#                     sim_vec.append(la.norm(in_sim_vec, l_norm))
#
#                 sim[sim_index] = sim_vec
#                 sim_index += 1
#
#                 print(sim_vec)
#
#
#             w_input.seek(0)
#             print("************************************************************")
#
#
#     # clf = svm.SVC(kernel='linear', C=1)
#     # scores = cross_validation.cross_val_score(clf, sim, target, cv = cv)
#     # return scores
#
# def cal_dist(list1, list2, beta, metric, window = 0, symmetry = False ,smooth = False):
#
#     min_len = min(len(list1), len(list2))
#
#     if metric == "dtw":
#         interval_num = min_len // beta
#         dist_vec = []
#
#         for i in range(interval_num):
#             begin = i * beta
#             end = begin + beta
#
#             interval_list1 = normalization(list1[begin:end])
#             interval_list2 = normalization(list2[begin:end])
#
#             dist_vec.append(dtw_wdistance(interval_list1, interval_list2, window, symmetry))
#
#     elif metric == "ddtw":
#         interval_num = min_len // beta
#         dist_vec = []
#
#         for i in range(interval_num):
#             begin = i * beta
#             end = begin + beta
#
#             interval_list1 = normalization(list1[begin:end])
#             interval_list2 = normalization(list2[begin:end])
#
#             dist_vec.append(ddtw_wdistance(interval_list1, interval_list2, window, symmetry, smooth))
#
#     return dist_vec
#
# def sim_write():
#     alpha_beta = [[3600, [12, 24, 48]], [86400, [7, 14, 30]], [604800, [4, 12, 24]], [2592000, [6, 12]]]
#     metric_sym_smo = [['dtw', False, False], ['dtw', True, False], ['ddtw', False, False], ['ddtw', False, True], ['ddtw', True, False], ['ddtw', True, True]]
#     bhv_cnts = [100, 300, 1000, 10000]
#
#     with open(zhihu_path, 'r') as zhihu, open(weibo_path, 'r') as weibo:
#         user_num = sum(1 for x in zhihu)
#         zhihu.seek(0)
#
#         for bhv_cnt in bhv_cnts:
#             user_dict = bhv_cnt_filter(bhv_cnt, zhihu_path, weibo_path)
#             print("bhv_cnt:%d" %bhv_cnt)
#
#             for alpha, betas in alpha_beta:
#                 print("alpha:%d" %alpha)
#                 sim = np.zeros((len(metric_sym_smo), len(betas), user_num, user_num))
#
#                 for z_user in zhihu:
#                     z_user = eval(z_user)
#
#                     if z_user['index'] not in user_dict:
#                         continue
#
#                     for w_user in weibo:
#
#                         w_user = eval(w_user)
#
#                         if w_user['index'] not in user_dict:
#                             continue
#
#                         z_list = Stamps(z_user['activity']).contiIntervalCnt(begin = 1458576000, interval = alpha)
#                         w_list = Stamps(w_user['activity']).contiIntervalCnt(begin = 1458576000, interval = alpha)
#
#                         for i in range(len(metric_sym_smo)):
#                             metric,symmetry,smooth = metric_sym_smo[i]
#                             sim_vec = cal_sim(z_list, w_list, betas, metric, symmetry = symmetry, smooth = smooth)
#                             for j in range(len(sim_vec)):
#                                 sim[i][j][z_user['index']][w_user['index']] = sim_vec[j]
#
#                     weibo.seek(0)
#                 zhihu.seek(0)
#
#                 out_path = "sim/alpha_%d_bhv_%d" %(alpha,bhv_cnt)
#                 with open(out_path, 'w+') as output:
#                     output.write(str(sim))
#
# def rank_write():
#     alpha_beta = [[3600, [12, 24, 48]], [86400, [7, 14, 30]], [604800, [4, 12, 24]], [2592000, [6, 12]]]
#     metric_sym_smo = [['dtw', False, False], ['dtw', True, False], ['ddtw', False, False], ['ddtw', False, True], ['ddtw', True, False], ['ddtw', True, True]]
#     bhv_cnts = [100, 300, 1000, 10000]
#     alpha_key = ['1hour', '1day', '1week', '1month']
#
#     with open(zhihu_path, 'r') as zhihu:
#         user_num = sum(1 for x in zhihu)
#
#     rank_dict = {}
#
#     for bhv_cnt in bhv_cnts:
#         for k,alpha,betas in enumerate(alpha_beta):
#             sim_path = "sim/bhv_%d_alpha_%d"%(bhv_cnt,alpha)
#             with open(sim_path, 'r') as sim:
#                 sim_mat = eval(sim.readline())
#
#             for i in range(len(metric_sym_smo)):
#                 for j in range(len(betas)):
#                     rank_cnt = rank_distribution(sim_mat[i][j])
#
#                     key = "%d_%s_%d_%s"%(bhv_cnt, alpha_key[k], betas[j])
#                     if metric_sym_smo[i][0] == 'dtw':
#                         key += '_dwt'
#                         if metric_sym_smo[i][1] == False:
#                             key += '_nsym'
#                         else:
#                             key += '_sym'
#                     else:
#                         key += '_ddwt'
#                         if metric_sym_smo[i][1] == False:
#                             key += '_nsym'
#                         else:
#                             key += '_sym'
#                         if metric_sym_smo[i][2] == False:
#                             key += '_nsmo'
#                         else:
#                             key += '_smo'
#
#                     rank_dict[key] = rank_cnt
#
#     rank_path = 'result/rank_distribution'
#     with open(rank_path, 'w+') as output:
#         output.write(str(ran_dict))
#
#
# def rank_distribution(sim_mat):
#     row, col = sim_mat.shape
#
#     row_rank_cnt = [0 for i in range(col)]
#     col_rank_cnt = [0 for i in range(row)]
#
#     for i in range(row):
#         pos_sim = sim_mat[i][i]
#         sorted_sim = sorted(sim_mat[i], reverse = True)
#         for j in range(len(sorted_sim)):
#             if sorted_sim[j] == pos_sim:
#                 row_rank_cnt[j] += 1
#                 break
#
#     for i in range(col):
#         pos_sim = sim_mat[i][i]
#         sorted_sim = sorted(sim_mat[:][i], reverse = True)
#         for j in range(len(sorted_sim)):
#             if sorted_sim[j] == pos_sim:
#                 col_rank_cnt[j] += 1
#                 break
#
#     return [row_rank_cnt, col_rank_cnt]
#
# def rank_plot():
#     with open(zhihu_path, 'r') as zhihu:
#         user_num = sum(1 for x in zhihu)
#
#     rank_path = 'result/rank_distribution'
#
#     with open(rank_path, 'r') as rank:
#         rank_dict = eval(rank.readline())
#
#     keys = ['100_1hour_12_dtw_nsym']
#
#     for key in keys:
#
#         plt.plot(range(1, user_num+1),rank_dict[key][0], color = 'blue',label='zhihu_'+key)
#         plt.plot(range(1, user_num+1),rank_dict[key][1], color = 'red',label='weibo_'+key)
#
#     ax = plt.gca()
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.spines['bottom'].set_position(('data',0))
#     ax.yaxis.set_ticks_position('left')
#     ax.spines['left'].set_position(('data',0))
#
#     plt.legend(loc='upper left')
#
#     plt.show()
#
#
#
# def filter_test():
#     test_user = range(10)
#     with open('data/zhihu/users.json', 'r') as input, open('data/zhihu/users_test.json', 'w') as output:
#         for line in input:
#             line = eval(line)
#             if line['index'] in test_user:
#                 output.write(str(line) + '\n')

#计算两条序列以beta分割的距离向量
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

        dist_vec.append(regular_euclid_distance(interval_list1, interval_list2))

    return dist_vec

def dist_in(beta, list1, list2):
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
        print(interval_list1, interval_list2)

        print(regular_euclid_distance(interval_list1, interval_list2))
        hh = input()

def dist_insight():
    options = [['day',7]]

    for option in options:
        print(option)

        z_cnt_path = 'data/zhihu/norm_cnt/{0}_series'.format(option[0])
        w_cnt_path = 'data/weibo/norm_cnt/{0}_series'.format(option[0])

        z_users = {}
        w_users = {}

        with open(z_cnt_path, 'r') as zhihu:
            for line in zhihu:
                line = eval(line)
                z_users[line['index']] = line['count']

        with open(w_cnt_path, 'r') as weibo:
            for line in weibo:
                line = eval(line)
                w_users[line['index']] = line['count']

        #输出文件名
        beta = option[1]

        dist_mat = zeros((1356, 1356), dtype = list)
        finished_cnt = 0
        start = time.time()
        for z_user in z_users:
            for w_user in w_users:
                if w_user == 7:
                    print(z_user, w_user)
                    dist_in(beta, z_users[z_user], w_users[w_user])



# 将用户序列间的距离矩阵写入文件
def series_dist_generator():
    options = [[7, 'hour'], [14, 'hour'], [14, 'weekday'], [30, 'hour'], [30, 'weekday']]

    for option in options:
        print(option)

        z_cnt_path = 'data/zhihu/norm_cnt/{0}_days_{1}'.format(option[0], option[1])
        w_cnt_path = 'data/weibo/norm_cnt/{0}_days_{1}'.format(option[0], option[1])

        out_path = 'result/division_statistic_dist/{0}_days_{1}'.format(option[0], option[1])

        z_users = {}
        w_users = {}

        with open(z_cnt_path, 'r') as zhihu:
            for line in zhihu:
                line = eval(line)
                z_users[line['index']] = line['series']

        with open(w_cnt_path, 'r') as weibo:
            for line in weibo:
                line = eval(line)
                w_users[line['index']] = line['series']

        #输出文件名
        if option[1] == 'hour':
            beta = 24
        elif option[1] == 'day':
            beta = 7

        dist_mat = zeros((1356, 1356), dtype = list)
        finished_cnt = 0
        start = time.time()
        for z_user in z_users:
            print(z_user)
            pool = multiprocessing.Pool(20)
            result_list = [x.get() for x in [pool.apply_async(cal_dist_vec,(beta, z_users[z_user], w_users[w_user],)) for w_user in w_users]]

            pool.close()
            pool.join()

            for (w_user,result) in enumerate(result_list):
                dist_mat[z_user][w_user] = result

            #print(dist_mat[z_user][:])
            with open(out_path, 'a') as output:
                output.write(str(ndarray.tolist(dist_mat[z_user][:])) + '\n')

            finished_cnt += 1
            print('%d users finished cost %f seconds'%(finished_cnt,time.time() - start))

        #print(dist_mat)



        #print(load(file_name))

#将用户的统计间的距离矩阵写入文件
def statistic_dist_generator():
    alphas = {'hour'}

    for alpha in alphas:
        z_path = 'data/zhihu/norm_cnt/%s_statistics'%alpha
        w_path = 'data/weibo/norm_cnt/%s_statistics'%alpha

        out_path = 'result/dist_mat/smooth_dtw_der/%s_dist_mat.txt'%alpha

        dist_mat = np.zeros((1356,1356))

        with open(z_path, 'r') as zhihu, open(w_path, 'r') as weibo:
            for z_line in zhihu:
                z_line = eval(z_line)
                for w_line in weibo:
                    w_line = eval(w_line)
                    dist_mat[z_line['index']][w_line['index']] = ddtw_wdistance(z_line['count'], w_line['count'], 2,smooth = True)

                weibo.seek(0)

        np.savetxt(out_path, dist_mat)

#在统计的距离矩阵中，找出同一个用户与自己是第几相似
def figure_identical_rank():
    dist_metries = ['derivate_euclid', 'euclid', 'smooth_derivate_euclid', 'dtw', 'dtw_der', 'smooth_dtw_der']
    alphas = ['hour']

    cnts = {1000}

    for cnt in cnts:
        users_index = bhv_cnt_filter(cnt)
        user_num = len(users_index)
        show_num = user_num // 10

        for dist_metry in dist_metries:
            for alpha in alphas:
                print(dist_metry, alpha)

                matrix_path = 'result/dist_mat/{0}/{1}_dist_mat.txt'.format(dist_metry, alpha)
                #matrix_path = 'result/match_mat/day_series_beta_7.txt'

                full_mat = np.loadtxt(matrix_path)

                dist_mat = np.zeros((user_num, user_num))

                for (x1, y1) in enumerate(users_index):
                    for (x2, y2) in enumerate(users_index):
                        dist_mat[x1][x2] = full_mat[y1][y2]

                K_distribution = np.zeros((2, user_num), dtype=np.float)

                # assignment
                for i in range(user_num):
                    row_k = sorted(dist_mat[i]).index(dist_mat[i][i])
                    K_distribution[0][row_k] += 1

                    col_k = sorted(dist_mat[:, i]).index(dist_mat[i][i])
                    K_distribution[1][col_k] += 1


                for i in range(2):
                    for j in range(1,user_num):
                        K_distribution[i][j] = K_distribution[i][j-1] + K_distribution[i][j]
                    for j in range(0,user_num):
                        K_distribution[i][j] = K_distribution[i][j] / user_num
                    print(K_distribution[0][0:show_num])

                # plot X-rows figure
                plt.clf()
                X = np.linspace(1, show_num, show_num)

                # plt.subplot(2, 1, 1)

                plt.plot(X, K_distribution[0][0:show_num])
                plt.xlim(1.0, float(show_num))

                # plt.subplot(2, 1, 2)
                # plt.plot(X, K_distribution[1][0:show_num])
                #
                # plt.xlim(1.0, float(show_num))

                title = '{0}_{1}_{2}'.format(alpha, dist_metry, cnt)
                plt.title(title)

                plt.ylabel('accuracy')
                plt.xlabel('k')
                plt.axis([1, show_num, 0, 1])

                path = 'figure/accuracy_k/{0}.png'.format(title)
                plt.savefig(path)

#plan A 将距离向量的矩阵写成match矩阵
def vec_to_match():
    options = [[7, 'hour'], [14, 'hour'], [14, 'weekday'], [30, 'hour'], [30, 'weekday']]

    for delta in [i/10 for i in range(1, 26)]:
        print(delta)
        for option in options:
            print(option)
            input_path = 'result/division_statistic_dist/{0}_days_{1}'.format(option[0], option[1])
            output_path = 'result/division_statistic_match_mat/{0}_days_{1}_delta_{2}.txt'.format(option[0], option[1], delta)
            match_mat = np.zeros((1356,1356))
            with open(input_path, 'r') as vec_mat:
                start = time.time()
                for (i,line) in enumerate(vec_mat):
                    line = np.array(eval(line))
                    for (j,vec) in enumerate(line):
                        vec = np.array(vec)
                        for dis in vec:
                            if dis <= delta:
                                match_mat[i][j] += 1
                        if len(vec) != 0:
                            match_mat[i][j] = match_mat[i][j] / len(vec)
                        else:
                            match_mat[i][j] = 0
                print(time.time() - start)
            np.savetxt(output_path, match_mat)

#根据距离矩阵选取前k个用户作为预测结果 计算准确率
def top_k_accuracy():
    cnt = 1000
    ks = [1,25,50,100]

    options = [[7, 'hour'], [14, 'hour'], [14, 'weekday'], [30, 'hour'], [30, 'weekday']]

    users_index = bhv_cnt_filter(cnt)
    user_num = len(users_index)

    acc_dict = {}

    for option in options:
        print(option)
        k_list = []
        for k in ks:
            print(k)
            delta_list = []
            for delta in [i/10 for i in range(1,21)]:
                print(delta)
                mat_path = 'result/division_statistic_match_mat/{0}_days_{1}_delta_{2}.txt'.format(option[0], option[1], delta)

                full_matrix = np.loadtxt(mat_path)

                matrix = np.zeros((user_num, user_num))
                for (x1, x2) in enumerate(users_index):
                    for (y1, y2) in enumerate(users_index):
                        matrix[x1][y1] = full_matrix[x2][y2]

                identify_correctly = 0

                for i in range(user_num):
                    self_value = matrix[i][i]
                    false_pos = 0
                    same_cnt = 0
                    for j in range(user_num):
                        # if self_value > matrix[i][j]:
                        #     false_pos += 1
                        if self_value < matrix[i][j]:
                            false_pos += 1
                        elif self_value == matrix[i][j]:
                            same_cnt += 1
                        if false_pos + same_cnt > k + 5:
                            break
                    if false_pos + same_cnt <= k + 5:
                        identify_correctly += 1

                accuracy = identify_correctly / user_num

                delta_list.append(accuracy)

            k_list.append(delta_list)

        acc_dict[str(option)] = k_list

    with open('result/accuracy/division_delta', 'w') as output:
        output.write(str(acc_dict))


# 根据距离矩阵 找出小于阈值的配对作为预测结果 计算准确率
def threshold_accuracy():
    cnt = 1000
    threshold = 1.2

    mat_path = 'result/dist_mat/euclid/hour_dist_mat.txt'

    full_matrix = np.loadtxt(mat_path)

    users_index = bhv_cnt_filter(cnt)
    user_num = len(users_index)

    matrix = np.zeros((user_num, user_num))
    for (x1, x2) in enumerate(users_index):
        for (y1, y2) in enumerate(users_index):
            matrix[x1][y1] = full_matrix[x2][y2]

    total = 0
    correct = 0

    for i in range(user_num):
        for j in range(user_num):
            if matrix[i][j] <= threshold:
                total += 1
                statistic_plot(users_index[i], users_index[j])
                if i == j:
                    correct += 1

    print(total, correct, correct/total)

# 画出距离矩阵中距离较小的用户对及相同用户
def statistic_plot(index1, index2):
    z_path = 'data/zhihu/norm_cnt/hour_statistics'
    w_path = 'data/weibo/norm_cnt/hour_statistics'

    z_user = {}
    w_user = {}

    with open(z_path, 'r') as zhihu, open(w_path, 'r') as weibo:
        for line in zhihu:
            line = eval(line)
            z_user[line['index']] = line['count']

        for line in weibo:
            line = eval(line)
            w_user[line['index']] = line['count']

    x = range(24)
    y1_zhihu = np.array(z_user[index1])
    y2_weibo = np.array(w_user[index2])
    y1_weibo = np.array(w_user[index1])
    y2_zhihu = np.array(z_user[index2])

    plt.plot(x, y1_zhihu, 'b-', label='user%d_zhihu'%index1)
    plt.plot(x, y2_weibo, 'r-', label='user%d_weibo'%index2)
    plt.plot(x, y1_weibo, 'g-', label='user%d_weibo'%index1)
    # plt.plot(x, y2_zhihu, 'r--', label='user%d_zhihu'%index2)

    plt.legend(loc='lower right')
    plt.show()

# 计算用户名之间的相似度
def cal_name_sim():
    z_name_path = 'data/zhihu/user_name'
    w_name_path = 'data/weibo/user_name'

    sim_mat = np.zeros((1356,1356))

    with open(z_name_path, 'r') as zhihu, open(w_name_path, 'r') as weibo:
        z_dict = eval(zhihu.readline())
        w_dict = eval(weibo.readline())

        for z_index in z_dict:
            z_name = z_dict[z_index]
            print(len(z_name))
            print(z_name)
            for w_index in w_dict:
                w_name = w_dict[w_index]
                if z_name != '' and w_name != '':
                    sim_mat[z_index][w_index] = 2 * LCSubstring(w_name, z_name) / (len(w_name + z_name))

    out_path = 'result/name_sim_mat/lcsubstr.txt'
    np.savetxt(out_path, sim_mat)

# 画准确率随delta变化的图
def plot_accuracy_delta():
    input_path = 'result/accuracy/division_delta'
    ks = [1, 25, 50, 100]

    with open(input_path, 'r') as accuracy:
        acc_dict = eval(accuracy.readline())

    for option in acc_dict:
        k_list = acc_dict[option]
        option = eval(option)

        plt.clf()

        title = '{0}_{1}'.format(option[0], option[1])
        plt.title(title)

        plt.ylabel('precision')

        save_path = 'figure/division_accuracy/{0}_{1}'.format(option[0], option[1])

        for (index, k) in enumerate(ks):
            X = np.linspace(0.1, 2.0, 20)
            Y = k_list[index]
            label = 'k=%s'%k
            plt.plot(X, Y, label=label)

        plt.legend(loc='upper left')

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data',0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data',0))

        plt.axis([0, 2.0, 0, 1])

        plt.savefig(save_path)

# 将hour_series 转化成每行为一天所有用户的24小时分布
# def series_day_extract():
#     input_path = 'data/weibo/norm_cnt/hour_series.txt'
#     output_path = 'data/weibo/norm_cnt/hour_series_day_extraction.txt'
#
#     users = np.loadtxt(input_path)
#     num_users = users.shape[0]
#     num_hours = 24
#     num_days = users.shape[1] // num_hours
#
#     matrix = np.zeros((num_users, num_days, num_hours))
#
#     start = time.time()
#     for index in range(num_users):
#         print(index)
#         series = users[index,:]
#         for day in range(1):
#             start_hour = day * 24
#             for h in range(24):
#                 matrix[index][day][h] = series[0,start_hour + h]
#     print(time.time() - start)
#
#     np.reshape(matrix, (1356, num_hours * num_days))
#     np.savetxt(output_path, matrix)

# 从得到的所有用户24小时分布上计算每小时的权重
def weight_calculation(dis_array):
    interval_num = 5

    probability_array = np.linspace(1, 0, interval_num + 1, True)
    interval_bins = stats.norm.isf(probability_array)

    entropy = np.zeros((24))

    num_user = dis_array.shape[0]
    num_hour = dis_array.shape[1]

    for h in range(num_hour):
        histogram = np.histogram(dis_array[:, h], interval_bins)[0]
        histogram = histogram / histogram.sum()
        entropy[h] = stats.entropy(histogram)

    entropy_sigma = np.sum(entropy)

    weight = np.zeros((24))

    if entropy_sigma == 0:
        for h in range(num_hour):
            weight[h] = 1 / 24
    else:
        for h in range(num_hour):
            weight[h] = entropy[h] / entropy_sigma

    return weight

# 采取寻找差异性天的方法,计算准确率
def day_sample_accuracy():
    target_path = 'data/zhihu/norm_cnt/hour_series_1000.txt'
    candidate_path = 'data/weibo/norm_cnt/hour_series_1000.txt'

    target_matrix = np.loadtxt(target_path)
    num_users = target_matrix.shape[0]

    num_days = 2411
    num_hours = 24

    candidate_matrix = np.loadtxt(candidate_path)
    candidate_matrix = np.reshape(candidate_matrix, (candidate_matrix.shape[0], num_days, num_hours))

    num_k = 16
    k_set = list(range(0, num_k * 10, 10))
    k_set[0] = 1

    atp = btp = np.zeros((num_k))

    for index in range(num_users):
        print(index)
        target_array = target_matrix[index,:]
        target_array = np.reshape(target_array, (num_days, num_hours))
        result = day_selection_identify(index, target_array, candidate_matrix, k_set)
        for i in range(num_k):
            if result[0][i] == True:
                atp[i] += 1
            if result[1][i] == True:
                btp[i] += 1

    A_acc = atp / num_users
    B_acc = btp / num_users

    plt.plot(k_set, A_acc, color='red', label='sum')
    plt.plot(k_set, B_acc, color='blue', label='vote')
    plt.legend(loc='upper left')
    plt.show()


# 2016/06.03
def pick_from_all_day_accuracy():
    pick_days_sum_set = list(range(100, 1201, 50))  # length=23
    k_set = list(range(0, 51, 5))  # length=11
    k_set[0] = 1
    A_result = np.zeros((len(pick_days_sum_set), len(k_set)))
    B_result = np.zeros((len(pick_days_sum_set), len(k_set)))

    target_path = 'data/zhihu/norm_cnt/hour_series_1000.txt'
    candidate_path = 'data/weibo/norm_cnt/hour_series_1000.txt'

    target_matrix = np.loadtxt(target_path)
    num_users = target_matrix.shape[0]  # target 的用户个数

    num_days = 2411
    num_hours = 24

    candidate_matrix = np.loadtxt(candidate_path)
    num_candidate = candidate_matrix.shape[0]
    candidate_matrix = np.reshape(candidate_matrix, (num_candidate, num_days, num_hours))

    for index in range(num_users):
        print(index)  # 每一个 target
        target_array = target_matrix[index, :]
        target_array = np.reshape(target_array, (num_days, num_hours))

        # target_array: 为A平台上用户u的时间序列表示 2411 x 24
        # candidate_matrix: B平台上所有用户的时间序列表示的集合 496 x 2411 x 24
        # 计算得到A平台上的target用户与所有其他用户在2411天上的距离矩阵，2411 x 496
        # dist_set 2411 x 496
        # entropy_set 2411
        dist_set, entropy_set = calculate_distance_and_entropy(target_array, candidate_matrix)

        for day_index, day_sum in enumerate(pick_days_sum_set):
            pick_days_index = get_top_k_index(entropy_set, day_sum)
            # day_sum x 496

            dist_set_of_pick_days = np.array([dist_set[day] for day in pick_days_index])
            entropy_set_of_pick_days = np.array([entropy_set[day] for day in pick_days_index])

            for k_index, k in enumerate(k_set):
                num = k
                if plan_A_value(dist_set_of_pick_days, entropy_set_of_pick_days, k, index):
                    A_result[day_index][k_index] += 1.0

                if plan_B_value(dist_set_of_pick_days, entropy_set_of_pick_days, num, k, index):
                    B_result[day_index][k_index] += 1.0

    A_acc = A_result / num_users
    B_acc = B_result / num_users

    A_result_path = "data/A_result.txt"
    B_result_path = "data/B_result.txt"

    np.savetxt(A_result_path, A_acc)
    np.savetxt(B_result_path, B_acc)

    # 观察在选择不同的well-days的天数时，这里就要固定置信区间，其准确率的变化图像
    for cols in range(len(k_set)):
        plt.plot(pick_days_sum_set, A_acc[:, cols], color='red', label='sum')
        plt.plot(pick_days_sum_set, B_acc[:, cols], color='blue', label='vote')
        plt.legend(loc='upper left')
        plt.show()

    # 观察在选择不同的置信区间时，即不同的k值，这里我们要固定所选择的well-days的天数，其准确率的变化图像
    for rows in range(len(pick_days_sum_set)):
        plt.plot(k_set, A_acc[rows, :], color='red', label='sum')
        plt.plot(k_set, B_acc[rows, :], color='blue', label='vote')
        plt.legend(loc='upper left')
        plt.show()


def calculate_distance_and_entropy(target_array, candidate_matrix):
    dist_set = []
    entropy_set = []
    num_of_all_days = target_array.shape[0]
    for day in range(num_of_all_days):
        target_day_series = target_array[day, :]
        candidate_day_series_matrix = candidate_matrix[:, day, :]  # 496 x 24

        hours_weight_in_day = weight_calculation(candidate_day_series_matrix)

        dist = distance_cal(target_day_series, candidate_day_series_matrix, hours_weight_in_day)
        dist_set.append(dist)

        # 计算在这一天day上所有的距离的熵,并记录下来
        max_dist = np.max(dist)
        min_dist = np.min(dist)
        num_bins = 50
        interval_bins = np.linspace(min_dist, max_dist, num_bins + 1, endpoint=True)
        histogram = np.histogram(dist, interval_bins)[0]
        histogram = histogram / histogram.sum()
        entropy_set.append(stats.entropy(histogram))

    return (np.array(dist_set), np.array(entropy_set))


# 给定一个用户，抽样n个差异性大于一定阈值的天， 计算准确度
def day_selection_identify(index, target_array, candidate_matrix, k_set):

    well_day_cnt = 30
    dist_set = []

    dataMat = list(range(2411))
    en_set = []
    while(well_day_cnt > 0):

        if len(dataMat) == 0:
            break
        selected_day = random.sample(dataMat, 1)
        selected_day = selected_day[0]
        dataMat.remove(selected_day)

        target_day_series = target_array[selected_day, :]

        w = weight_calculation(candidate_matrix[:, selected_day, :])
        dist = distance_cal(target_day_series, candidate_matrix[:, selected_day, :], w)

        max_dist = np.max(dist)
        min_dist = np.min(dist)

        num_bins = 10

        interval_bins = np.linspace(min_dist, max_dist, num_bins + 1, endpoint=True)

        histogram = np.histogram(dist, interval_bins)[0]
        histogram = histogram / histogram.sum()
        entropy = stats.entropy(histogram)

        deta = 1.3

        if entropy > deta:
            en_set.append(entropy)
            well_day_cnt -= 1
            dist_set.append(dist)

    # plt.hist(np.array(en_set), 50, color ='red', alpha = 0.5)
    # plt.show()
    result = [[] for i in range(2)]

    for k in k_set:
        num = k
        dist_set = np.array(dist_set)
        en_set = np.array(en_set)

        result[0].append(plan_A_value(dist_set, en_set, k, index))
        result[1].append(plan_B_value(dist_set, en_set, num, k, index))

    return result


def distance_cal(target, candidates, w):
    '''
    :target list len=24 某一用户在一天内的时间序列表示
    :candidates np.array((m, 24)) 表示候选集合中的所有用户在这一天上的时间序列表示
    :w list len=24 表示用户在这一天24小时的每个小时上的不同权重
    '''
    Dist = []
    for candidate in candidates:
        dist = 0
        for h in range(24):
            dist += w[h] * (target[h] - candidate[h])**2
        Dist.append(sqrt(dist))
    return np.array(Dist)


def get_top_k_index(lvalue, k):
    '''
    :para
    :lvalue a list that contains values
    :k top-k
    :return
    :top_k_index the index of all items in top-k
    '''
    lvalue_index = [{'index': index, 'value': value} for index, value in enumerate(lvalue)]
    top_k = nlargest(k, lvalue_index, key=lambda item: item['value'])
    top_k_index = [each_item['index'] for each_item in top_k]
    return top_k_index


def plan_A_sum(D, k):
    '''
    :D np.array((n, m)) 距离矩阵
    :k top-k
    '''
    # num_of_users = D.shape[1]
    day_sum_list = D.sum(axis=0)
    top_k_sum_index = get_top_k_index(day_sum_list, k)
    return top_k_sum_index


def plan_A_value(D, entropy, k, target_index):
    D = np.dot(entropy, D)
    distinct_sum_set = set(nlargest(k, D))
    return (D[target_index] in distinct_sum_set)


def plan_B_voting(D, num, k):
    '''
    :D np.array((n, m)) 距离矩阵
    :num 每一行的前 num 个距离对应为1，否则为0
    :k top-k
    '''
    # n, m = D.shape
    match = np.zeros(D.shape)
    for index, line in enumerate(D):
        for i in get_top_k_index(line, num):
            match[index][i] = 1.0
    voting_sum = match.sum(axis=0)
    plt.hist(voting_sum, 30, color='red', alpha=0.5)
    plt.show()
    top_k_voting_index = get_top_k_index(voting_sum, k)
    return top_k_voting_index


def plan_B_value(D, entropy, num, k, target_index):
    match = np.zeros(D.shape)
    for index, line in enumerate(D):
        for i in get_top_k_index(line, num):
            match[index][i] = 1.0
    voting_sum = np.dot(entropy, match)
    distinct_voting_set = set(nlargest(k, voting_sum))
    return (voting_sum[target_index] in distinct_voting_set)


# 过滤出行为数大于1000的用户生成矩阵
def filter_users():
    platforms = ['zhihu', 'weibo']

    bhv_threshold = 1000
    satisfied_index = bhv_cnt_filter(bhv_threshold)

    for platform in platforms:
        input_path = 'data/{0}/norm_cnt/hour_series.txt'.format(platform)
        output_path = 'data/{0}/norm_cnt/hour_series_{1}.txt'.format(platform, bhv_threshold)

        in_matrix = np.loadtxt(input_path)
        out_matrix = np.zeros((len(satisfied_index), in_matrix.shape[1]))

        out_index = 0

        for in_index in satisfied_index:
            out_matrix[out_index,:] = in_matrix[in_index]
            out_index += 1

        np.savetxt(output_path, out_matrix)




def main():
    # top_k_accuracy()
    # vec_to_match()
    # threshold_accuracy()
    # cal_name_sim()
    # statistic_dist_generator()
    # figure_identical_rank()
    # plot_accuracy_delta()
    # series_dist_generator()
    # dist_insight()
    # series_day_extract()
    # day_sample_accuracy()
    # filter_users()
    pick_from_all_day_accuracy()

if __name__ == "__main__":
    main()
