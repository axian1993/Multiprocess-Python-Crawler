# -*- coding: utf-8 -*-

# Requirements
import requests

# Module
from weibo import User

def crawl_id():
    input_path = 'data/available_users'

    w_id = {}

    with open(input_path, 'r') as users:
        for user in users:
            user = eval(user)
            w_user = User(user['weibo_link'])

            try:
                w_id[user['index']] = w_user.get_user_name()
                print(user['index'])
            except:
                print(user['index'], w_user[user['index']])


    output_path = 'data/user_name'

    with open(output_path, 'w') as out:
        out.write(str(w_id))

def find_empty_name():
    input_path = 'data/available'
    name_path = 'data/user_name'

    with open(name_path, 'r') as names:
        z_id = eval(names.readline())
        for index in z_id:
            if z_id[index] == '':
                print(index)

def main():
    find_empty_name()

if __name__ == '__main__':
    main()
