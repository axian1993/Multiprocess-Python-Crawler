# -*- coding: utf-8 -*-
import requests
import multiprocessing, os
from multiprocessing import Pool
from bs4 import BeautifulSoup

request = requests.Session()

def write_file(queue, write, lock):
    output = open('data/available_users', 'w')
    index = 0

    while True:
        try:
            write.wait()
            user = queue.get()
            if user == None:
                break
            user['index'] = index
            index += 1
            output.write(str(user) + '\n')
            write.clear()
            lock.release()
        except Exception as e:
            print(e)

    output.close()

def filter_user( id, queue, lock, write):
    print("filter pid: " + str(os.getpid()))
    url = 'http://www.zhihu.com/people/'
    user = {}
    try:
        print(url + id)
        r = request.get(url + id)
        soup = BeautifulSoup(r.content.decode('UTF-8'), "html.parser")
        if soup.find("a", class_="zm-profile-header-user-weibo"): #判断是否有微博超链接，若无则返回空字典
            user['id'] = id
            user['weibo_link'] = soup.find("a", class_="zm-profile-header-user-weibo")["href"]

            lock.acquire()
            queue.put(user)
            write.set()


    except Exception as e:
        print(e)
        print("pid: " + str(os.getpid()) + "failed: " + url + id)





if __name__ == "__main__":
    input = open('data/user_id_mixed.json', 'r')
    ids = eval(input.readline())['id']
    input.close()

    lock = multiprocessing.Manager().Lock()
    queue = multiprocessing.Manager().Queue()
    write = multiprocessing.Manager().Event()

    p = multiprocessing.Process(target = write_file, args = (queue, write, lock,))
    p.start()

    with multiprocessing.Pool(processes = 4) as pool:
        for id in ids:
            pool.apply_async(filter_user, (id, queue, lock, write,))

        pool.close()
        pool.join()

    write.set()
    queue.put(None)

    p.join()

    print("finished")
