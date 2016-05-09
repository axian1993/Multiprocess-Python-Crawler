# required module
import matplotlib
import numpy as np
import pylab as pl

matplotlib.use('Agg')

# import from similarity.py
from similarity import dtw_wdistance, ddtw_wdistance


# open a  log_file.txt, append log into it
log_path = '/Users/bing/Desktop/PostGra/experiment/data/static/log_file.txt'
logging = open(log_path, 'a')
print('successfully opening the log file')

'''
:para:stype(short for statistic), totally four types: 'hour','weekday','month','year'
:pare:symmtry-; smooth-
     :simi_index      calculate_similarity
        :0           dtw_wdistance(list1, list2, 0, symmetry=False)
        :1           dtw_wdistance(list1, list2, 0, symmetry=True)
        :2           ddtw_wdistance(list1, list2, 0, symmetry=False, smooth=False)
        :3           ddtw_wdistance(list1, list2, 0, symmetry=True, smooth=False)
        :4           ddtw_wdistance(list1, list2, 0, symmetry=False, smooth=True)
        :5           ddtw_wdistance(list1, list2, 0, symmetry=True, smooth=True)
'''

def statistic_similarity(stype, simi_index):

    zhihu_file_path = "/Users/bing/Desktop/PostGra/experiment/data/zhihu/norm_cnt/{0}_statistics".format(stype)
    weibo_file_path = "/Users/bing/Desktop/PostGra/experiment/data/weibo/norm_cnt/{0}_statistics".format(stype)

    similarity_matrix = np.zeros((1356, 1356))
    similarity_matrix_path = "/Users/bing/Desktop/PostGra/experiment/data/static/{0}_{1}_similarity_matrix.txt".format(stype, simi_index)

    # one user in zhihu -> all users in weibo, for first time
    # one target user -> many candidate users
    with open(zhihu_file_path, 'r') as zhihu, open(weibo_file_path, 'r') as weibo:
        for zh_line in zhihu:
            zh_line = eval(zh_line)
            target_index, target_list = zh_line['index'], zh_line['count']
            print(str(target_index))
            for wb_line in weibo:
                wb_line = eval(wb_line)
                candidate_index, candidate_list = wb_line['index'], wb_line['count']

                if simi_index == 0:
                    similarity_matrix[target_index][candidate_index] = dtw_wdistance(target_list, candidate_list, 0)
                elif simi_index == 1:
                    similarity_matrix[target_index][candidate_index] = dtw_wdistance(target_list, candidate_list, 0, symmetry=True)
                elif simi_index == 2:
                    similarity_matrix[target_index][candidate_index] = ddtw_wdistance(target_list, candidate_list, 0)
                elif simi_index == 3:
                    similarity_matrix[target_index][candidate_index] = ddtw_wdistance(target_list, candidate_list, 0, symmetry=True)
                elif simi_index == 4:
                    similarity_matrix[target_index][candidate_index] = ddtw_wdistance(target_list, candidate_list, 0, smooth=True)
                elif simi_index == 5:
                    similarity_matrix[target_index][candidate_index] = ddtw_wdistance(target_list, candidate_list, 0, symmetry=True, smooth=True)
            # caution!!!
            weibo.seek(0)

    np.savetxt(similarity_matrix_path, similarity_matrix, fmt='%.10f')
    logging.write(stype + '-' +str(simi_index) + "-similarity_matrix-done-----------------\n")

    return similarity_matrix


def figure_k_count(stype, method):
    simi_matrix = statistic_similarity(stype, method)

    K_distribution = np.zeros((2, 1356), dtype=np.int)

    # assignment
    for i in range(1356):
        row_k = sorted(simi_matrix[i]).index(simi_matrix[i][i])
        K_distribution[0][row_k] += 1

        col_k = sorted(simi_matrix[:, i]).index(simi_matrix[i][i])
        K_distribution[1][col_k] += 1

    k_distribution_path = '/Users/bing/Desktop/PostGra/experiment/data/static/k_distribution/{0}_{1}_k_distribute'.format(stype, method)
    np.savetxt(k_distribution_path, K_distribution, fmt='%d')
    #with open(k_distribution_path, 'w') as k_path:
    #    out = {}
    #    out['each-row-distribution'] = rows_K_distribution
    #    out['each-col-distribution'] = cols_K_distribution
    #    k_path.write(str(out) + '\n')
    logging.write("has writtint the distribution into {0} \n".format(k_distribution_path))

    # plot X-rows figure
    pl.clf()
    pl.figure(figsize=(30, 10), dpi=10)
    X = np.linspace(0, 1355, 1356)

    pl.subplot(2, 1, 1)
    pl.plot(X, K_distribution[0], 'r.')
    pl.ylabel('#row-cnt')
    title = '{0}-{1}-Similarity-K-Distributinon'.format(stype, method)
    pl.title(title)

    pl.subplot(2, 1, 2)
    pl.plot(X, K_distribution[1], 'b.')
    pl.ylabel('#col-cnt')

    pl.xlim(0.0, 1355.0)
    pl.xlabel('#Matching in K')

    path = '/Users/bing/Desktop/PostGra/experiment/data/static/{0}.png'.format(title)
    pl.savefig(path)
    logging.write("has save the {} image \n".format(path))


def main():
    stype_list = ['hour', 'weekday', 'month', 'year']
    for stype in stype_list:
        for i in range(6):
            figure_k_count(stype, i)
            logging.write('figure {0}_{1}_similarity, done!\n'.format(stype, i))
        logging.write('----------------calculate {0}_similarity, done!------------------\n'.format(stype))
        logging.write('------------------------------------------------\n')


if __name__ == '__main__':
    main()

