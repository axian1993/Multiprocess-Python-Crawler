# -*- coding: utf-8 -*-

def bhv_cnt_filter(cnt, path1, path2):
    z_index = []
    w_index = []
    common_index = {}
    with open(path1, 'r') as z_user, open(path2, 'r') as w_user:
        for line in z_user:
            line = eval(line)
            if len(line['activity']) >= cnt:
                z_index.append(line['index'])

        for line in w_user:
            line = eval(line)
            if len(line['activity']) >= cnt:
                w_index.append(line['index'])

        for i in z_index:
            if i in w_index:
                common_index[i] = 1

    return common_index

def main():
    bhv_cnt_filter(100)

if __name__ == '__main__':
    main()
