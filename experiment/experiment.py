# -*- coding: utf-8 -*-

#built-in/std
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import time
import multiprocessing

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
            pool = multiprocessing.Pool(2)
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
    options = [['day', 7], ['day', 14], ['day', 30], ['week', 4], ['week', 12], ['week', 24], ['month',6], ['month', 12]]

    for delta in [i/10 for i in range(1, 16)]:
        print(delta)
        for option in options:
            print(option)
            input_path = 'result/dist_vec_mat/{0}_series_beta_{1}.txt'.format(option[0], option[1])
            output_path = 'result/match_mat_delta/{0}_series_beta_{1}_delta_{2}.txt'.format(option[0], option[1], delta)
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

    options = [['day', 7], ['day', 14], ['day', 30], ['week', 4], ['week', 12], ['week', 24], ['month',6], ['month', 12]]

    users_index = bhv_cnt_filter(cnt)
    user_num = len(users_index)

    acc_dict = {}

    for option in options:
        print(option)
        k_list = []
        for k in ks:
            print(k)
            delta_list = []
            for delta in [i/10 for i in range(1,16)]:
                print(delta)
                mat_path = 'result/match_mat_delta/{0}_series_beta_{1}_delta_{2}.txt'.format(option[0], option[1], delta)

                full_matrix = np.loadtxt(mat_path)

                matrix = np.zeros((user_num, user_num))
                for (x1, x2) in enumerate(users_index):
                    for (y1, y2) in enumerate(users_index):
                        matrix[x1][y1] = full_matrix[x2][y2]

                identify_correctly = 0

                for i in range(user_num):
                    self_value = matrix[i][i]
                    false_pos = 0
                    for j in range(user_num):
                        # if self_value > matrix[i][j]:
                        #     false_pos += 1
                        if self_value < matrix[i][j]:
                            false_pos += 1
                        if false_pos >= k:
                            break
                    if false_pos < k:
                        identify_correctly += 1

                accuracy = identify_correctly / user_num

                delta_list.append(accuracy)

            k_list.append(delta_list)

        acc_dict[str(option)] = k_list

    with open('result/accuracy/series_delta', 'w') as output:
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
    input_path = 'result/accuracy/series_delta'
    ks = [1, 25, 50, 100]

    with open(input_path, 'r') as accuracy:
        acc_dict = eval(accuracy.readline())

    for option in acc_dict:
        k_list = acc_dict[option]
        option = eval(option)

        plt.clf()

        title = 'series_{0}_{1}'.format(option[0], option[1])
        plt.title(title)

        plt.xlabel('delta')
        plt.ylabel('accuracy')

        save_path = 'figure/series_accuracy/{0}_{1}'.format(option[0], option[1])

        for (index, k) in enumerate(ks):
            X = np.linspace(0.1, 1.5, 15)
            Y = k_list[index]
            label = 'k=%s'%k
            plt.plot(X, Y, label=label)

        plt.savefig(save_path)


def main():
    # top_k_accuracy()
    # vec_to_match()
    # threshold_accuracy()
    # cal_name_sim()
    # statistic_dist_generator()
    # figure_identical_rank()
    # plot_accuracy_delta()
    series_dist_generator()

if __name__ == "__main__":
    main()
