# -*- coding: utf-8 -*-
'''
test weibo.py by qiaoIn
'''

from weibo import User

def test_weibo():
    url = 'http://weibo.cn/bassix?page=1'
    user = User(url)
    print(user.get_user_id()) # 打印出 ID 号
    for timestamp in user.get_activities_timestamp():
        print(timestamp) # 打印时间戳

if __name__ == '__main__':
    test_weibo()
