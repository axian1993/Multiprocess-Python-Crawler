# -*- coding: utf-8 -*-

#module
from series_transfer import *
from experiment import *
from user_filter import *
import multiprocessing
import time

zhihu_path = 'data/zhihu/users.json'
weibo_path = 'data/weibo/users.json'

manager = multiprocessing.Manager()
write_lock = manager.Lock()

def cal_interval_sim(user1, user2, alpha, beta, output):
    sim_dict = {}

    sim_dict['user_pair'] = [user1['index'], user2['index']]
    sim_dict['sim'] = []

    metric_sym_smo = [['dtw', False, False], ['dtw', True, False], ['ddtw', False, False], ['ddtw', False, True], ['ddtw', True, False], ['ddtw', True, True]]

    z_list = Stamps(user1['activity']).contiIntervalCnt(begin = 1458576000, interval = alpha)
    w_list = Stamps(user2['activity']).contiIntervalCnt(begin = 1458576000, interval = alpha)

    for i in range(len(metric_sym_smo)):
        metric,symmetry,smooth = metric_sym_smo[i]
        #print(metric, symmetry, smooth)
        sim_vec = cal_sim(z_list, w_list, beta, metric, symmetry = symmetry, smooth = smooth)
        sim_dict['sim'].append(sim_vec)

    write_lock.acquire()
    output.write(str(sim_dict) + '\n')
    write_lock.release()

def main():
    alpha_beta = [[86400, [7, 14, 30]], [604800, [4, 12, 24]], [2592000, [6, 12]]]

    with open(zhihu_path, 'r') as zhihu, open(weibo_path, 'r') as weibo:
        for alpha,betas in alpha_beta:
            for beta in betas:
                result_path = 'result/interval_sim/alpha_%d_beta_%d'%(alpha, beta)
                print(alpha, beta)
                start_time = time.time()
                with open(result_path, 'w') as result_file:
                    pool = multiprocessing.Pool(20)
                    for z_user in zhihu:
                        z_user = eval(z_user)
                        for w_user in weibo:
                            w_user = eval(w_user)
                            pool.apply_async(cal_interval_sim,(z_user, w_user, alpha, beta, result_file,))
                        weibo.seek(0)
                    pool.close()
                    pool.join()
                    zhihu.seek(0)
                print(time.time() - start_time)



if __name__ == "__main__":
    main()
