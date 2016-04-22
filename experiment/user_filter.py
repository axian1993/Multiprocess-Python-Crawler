# -*- coding: utf-8 -*-

def bhv_cnt_filter(cnt):
    z_index = []
    w_index = []
    common_index = {}
    with open('data/zhihu/users_test.json') as z_user, open('data/weibo/users_test.json') as w_user:
        for line in z_user:
            line = eval(line)
            if len(line['activity']) >= 100:
                z_index.append(line['index'])

        for line in w_user:
            line = eval(line)
            if len(line['activity']) >= 100:
                w_index.append(line['index'])

        for i in z_index:
            if i in w_index:
                common_index[i] = 1

    return common_index

def main():
    bhv_cnt_filter(100)

if __name__ == '__main__':
    main()
