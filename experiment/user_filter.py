# -*- coding: utf-8 -*-

def bhv_cnt_filter(cnt):
    z_path = 'data/zhihu/bhv_cnt'
    w_path = 'data/weibo/bhv_cnt'

    z_index = []
    w_index = []
    common_index = []
    with open(z_path, 'r') as z_user, open(w_path, 'r') as w_user:
        for line in z_user:
            line = eval(line)
            if line['cnt'] >= cnt:
                z_index.append(line['index'])

        for line in w_user:
            line = eval(line)
            if line['cnt'] >= cnt:
                w_index.append(line['index'])

        for i in z_index:
            if i in w_index:
                common_index.append(i)

    return common_index

def cnt_behavior():
    z_path = 'data/zhihu/users.json'
    w_path = 'data/weibo/users.json'

    z_out_path = 'data/zhihu/bhv_cnt'
    w_out_path = 'data/weibo/bhv_cnt'

    with open(z_path, 'r') as zhihu, open(w_path, 'r') as weibo, open(z_out_path, 'w') as z_out, open(w_out_path, 'w') as w_out:
        for line in zhihu:
            line = eval(line)
            user = {}
            user['index'] = line['index']
            user['cnt'] = len(line['activity'])
            z_out.write(str(user) + '\n')

        for line in weibo:
            line = eval(line)
            user = {}
            user['index'] = line['index']
            user['cnt'] = len(line['activity'])
            w_out.write(str(user) + '\n')


def main():
    cnt_behavior()

if __name__ == '__main__':
    main()
