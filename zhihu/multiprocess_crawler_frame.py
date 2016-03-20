# -*- coding = utf-8 -*-

#Build-in / Std
import threading, multiprocessing, time, traceback, sys
import random

#Requirments
from bs4 import BeautifulSoup
import requests

#module
from zhihu import User

free_proxy_site_url = "http://www.ip84.com/gn/" #免费代理网站url
number_of_pages = 10    #单次爬取代理页数

manager = multiprocessing.Manager()

proxies_container = manager.list()
proxies_container.append([])
proxy_start_event = manager.Event()
proxy_apply_event = manager.Event()
proxy_recycle_event = manager.Event()
proxy_queue = manager.Queue()
proxy_lock = manager.Lock()


def get_proxies():
    request = requests.Session()

    proxies = []
    current_page = 0

    while current_page < number_of_pages:
        url = free_proxy_site_url + str(current_page)
        try:
            r = request.get(url)
            soup = BeautifulSoup(r.content, "html.parser")
            proxy_list = soup.find("table", class_ = "list").find_all("tr")

            for i in range(1, len(proxy_list)):
                row = proxy_list[i].find_all("td")

                if row[4].string == 'HTTPS':
                    proxy = row[0].string + ":" + row[1].string
                    proxies.append(proxy)

        except requests.exceptions.ConnectionError:
            print("connection error while getting proxy in page" + str(current_page + 1))
            time.sleep(3)

        except:
            print("other error while getting proxy in page" + str(current_page + 1))
            info = sys.exc_info()
            print(str(info[0]) + ':' + str(info[1]))
            current_page += 1

        else:
            current_page += 1

    print("proxy got")
    return proxies

def update_proxies():
    while True:
        time.sleep(600)
        proxies = get_proxies()
        proxies_container[0] = proxies

def proxy_apply_handler():

    while True:
        proxy_apply_event.wait()
        proxies = proxies_container[0]
        if not proxies:
            proxies = get_proxies()
        proxy = random.choice(proxies)
        proxy_queue.put(proxy)
        proxies.remove(proxy)
        proxies_container[0] = proxies

        proxy_apply_event.clear()
        proxy_lock.release()

def proxy_apply():
    proxy_lock.acquire()
    proxy_apply_event.set()
    proxy = proxy_queue.get()
    return proxy

def proxy_recycle_handler():
    while True:
        proxy_recycle_event.wait()
        proxy = proxy_queue.get()

        proxies = proxies_container[0]
        if proxy not in proxies:
            proxies.append(proxy)
        proxies_container[0] = proxies

        proxy_recycle_event.clear()
        proxy_lock.release()

def proxy_recycle(proxy):
    proxy_lock.acquire()
    proxy_queue.put(proxy)
    proxy_recycle_event.set()

def proxies_maintain():
    proxies_container[0] = get_proxies()

    thread_apply = threading.Thread(target = proxy_apply_handler)
    thread_recycle = threading.Thread(target = proxy_recycle_handler)
    thread_update = threading.Thread(target = update_proxies)

    thread_apply.setDaemon(True)
    thread_recycle.setDaemon(True)
    thread_update.setDaemon(True)

    thread_update.start()
    thread_apply.start()
    thread_recycle.start()

    proxy_start_event.set()

    while  True:
        pass

write_event = manager.Event()
write_queue = manager.Queue()
write_lock = manager.Lock()

def writer_get_info():
    write_event.wait()
    info = write_queue.get()

    write_event.clear()
    write_lock.release()

    return info

def pass_to_writer(info):
    write_lock.acquire()
    write_queue.put(info)
    write_event.set()

def writer():
    with open("data/users.json", 'w') as user, open("data/error_users.json", "w") as error_users:
        while True:
            info = writer_get_info()
            try:
                if info == None:
                    break
                elif info['error'] == '':
                    user.write(str(info) + "\n")
                else:
                    error_users.write(str(info) + "\n")
            except:
                print(info)
                traceback.print_exc()

def get_user_id(path):
    users = []
    with open(path, 'r') as input:
        for line in input:
            user = eval(line)
            users.append(user)

    return users

def crawler(user_info):
    url = "http://www.zhihu.com/people/" + user_info['id']

    try:
        user = User(url)

        proxy = proxy_apply()
        activities = []

        print("start crawl " + url + "\nproxy:" + proxy + '\n')

        for activity in user.get_activities(proxy):
            activities.append(eval(activity))
        user_info['activity'] = activities

        user_info['error'] = ''
        pass_to_writer(user_info)
        proxy_recycle(proxy)
        print(url + 'finished\n')

    except requests.exceptions.ConnectionError or requests.exceptions.ReadTimeout as e:
        print(url + " requests error:")
        print(e)
        print("changing the proxy.............\n")
        crawler(user_info)

    except Exception as e:
        info = sys.exc_info()
        user_info['error'] = str(info[0]) + ':' + str(info[1])
        pass_to_writer(user_info)

        print("other error")
        print(e)


if __name__ == "__main__":
    p = multiprocessing.Process(target = proxies_maintain)
    p.start()
    proxy_start_event.wait()

    writer_process = multiprocessing.Process(target = writer)
    writer_process.start()

    source_path = "data/users_test"
    users = get_user_id(source_path)

    start = time.time()

    pool = multiprocessing.Pool(32)
    for user in users:
        pool.apply_async(crawler, (user,))

    pool.close()
    pool.join()

    end = time.time()

    write_event.set()
    pass_to_writer(None)
    writer_process.join()

    p.terminate()

    print("finished\ttime cost: " + str(end - start))
