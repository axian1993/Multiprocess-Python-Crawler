# -*- coding: utf-8 -*-

# Requirements
import requests

# Module
from zhihu import User

def crawl_id():
    input_path = 'data/available_users'
    output_path = 'data/user_name'

    z_id = {}

    zhihu_url = 'http://www.zhihu.com/people/'

    with open(input_path, 'r') as users:
        for user in users:
            user = eval(user)
            z_user = User(zhihu_url + user['id'])
            try:
                z_id[user['index']] = z_user.get_user_id()
                print(user['index'])
            except:
                print(user['index'], z_user[user['index']])




    with open(output_path, 'w') as out:
        out.write(str(z_id))

def find_empty_name():
    input_path = 'data/available'
    name_path = 'data/user_name'

    with open(name_path, 'r') as names:
        z_id = eval(names.readline())
        for index in z_id:
            if z_id[index] == '':
                print(index)

def fill_name():
    name_path = 'data/user_name'

    index = 1170
    name = '米欧格斯'

    with open(name_path, 'r+') as names:
        z_id = eval(names.readline())
        z_id[index] = name
        names.seek(0)
        names.write(str(z_id))

def main():
    fill_name()

if __name__ == '__main__':
    main()
