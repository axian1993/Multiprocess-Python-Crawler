# -*- coding:utf-8 -*-

from zhihu import User
import time

if __name__ == '__main__':
    output = open('out_test', 'w')
    start = time.time()
    activities = User('http://www.zhihu.com/people/a-xian-4-10').get_activities('117.135.251.134:80')
    for activity in activities:
        output.write(activity + '\n')

    end = time.time()
    output.close()
    print("time taken:" + str(end - start) + 's')
