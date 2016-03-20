# -*- coding = utf-8 -*-

#Build-in / Std
import threading, multiprocessing, time, traceback, sys
import random

#Requirments
from bs4 import BeautifulSoup
import requests

#module
from zhihu import User

#免费代理网站url
free_proxy_site_url = "http://www.ip84.com/gn/"

#单次爬取代理页数
number_of_pages = 10

#并发爬虫进程数量
number_of_multiprocessing = 16

#服务多进程的管理进程
manager = multiprocessing.Manager() #

#存放代理池列表的容器
proxies_container = manager.list()
proxies_container.append([])

#等待代理进程启动完毕的Event
proxy_start_event = manager.Event()

#处理申请代理的Event
proxy_apply_event = manager.Event()

#回收代理的Event
proxy_recycle_event = manager.Event()

#爬虫与写入进程通信的队列
proxy_queue = manager.Queue()

#代理操作锁
proxy_lock = manager.Lock()

#代理池自动更新时间
proxy_pool_update_period = 3600.0

#user-agent池
user_agent_pool = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X MetaSr 1.0",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:16.0) Gecko/20121026 Firefox/16.0",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:2.0b13pre) Gecko/20110307 Firefox/4.0b13pre",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:16.0) Gecko/20100101 Firefox/16.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16",
    "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0)"
]

#爬取免费代理网站的代理地址
def get_proxies():
    request = requests.Session()

    proxies = []
    current_page = 1

    while current_page <= number_of_pages:
        url = free_proxy_site_url + str(current_page)
        try:
            r = request.get(url)
            soup = BeautifulSoup(r.content, "html.parser")
            proxy_list = soup.find("table", class_ = "list").find_all("tr")

            for i in range(1, len(proxy_list)):
                row = proxy_list[i].find_all("td")

                #仅爬取支持HTTPS协议的代理
                if row[4].string == 'HTTPS':
                    proxy = row[0].string + ":" + row[1].string
                    proxies.append(proxy)

        except requests.exceptions.ConnectionError as e:
            print("connection error while getting proxy in page" + str(current_page))
            print(e)
            time.sleep(3)

        except Exception as e:
            print("other error while getting proxy in page" + str(current_page))
            print(e)
            current_page += 1

        else:
            current_page += 1

    print("proxy got\n")
    return proxies

#更新代理池
def update_proxies():
    while True:
        time.sleep(proxy_pool_update_period)
        proxies = get_proxies()
        proxies_container[0] = proxies

#代理申请处理线程
def proxy_apply_handler():
    while True:
        proxy_apply_event.wait()
        proxies = proxies_container[0]

        #若代理池为空，则更新代理池
        if not proxies:
            proxies = get_proxies()

        #从代理池中随机抽样
        proxy = random.choice(proxies)
        proxy_queue.put(proxy)
        proxies.remove(proxy)
        proxies_container[0] = proxies

        proxy_apply_event.clear()
        proxy_lock.release()

#申请代理方法
def proxy_apply():
    proxy_lock.acquire()
    proxy_apply_event.set()
    proxy = proxy_queue.get()
    return proxy

#代理回收线程
def proxy_recycle_handler():
    while True:
        proxy_recycle_event.wait()
        proxy = proxy_queue.get()
        proxies = proxies_container[0]

        #如果代理不在代理池中，则将代理回收
        if proxy not in proxies:
            proxies.append(proxy)

        proxies_container[0] = proxies

        proxy_recycle_event.clear()
        proxy_lock.release()

#代理回收方法
def proxy_recycle(proxy):
    proxy_lock.acquire()
    proxy_queue.put(proxy)
    proxy_recycle_event.set()

#代理池维持进程
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

    #等待被主进程terminate
    while  True:
        pass

#写进程控制Event
write_event = manager.Event()

#爬虫与写进程通信队列
write_queue = manager.Queue()

#写进程锁
write_lock = manager.Lock()

#写进程获取爬虫信息方法
def writer_get_info():
    write_event.wait()
    info = write_queue.get()

    write_event.clear()
    write_lock.release()

    return info

#爬虫将信息传递给写进程方法
def pass_to_writer(info):
    write_lock.acquire()
    write_queue.put(info)
    write_event.set()

#写进程
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

#获取需要爬取用户的id
def get_user_id(path):
    users = []
    with open(path, 'r') as input:
        for line in input:
            user = eval(line)
            users.append(user)

    return users

#爬虫进程
def crawler(user_info):
    url = "http://www.zhihu.com/people/" + user_info['id']

    try:
        user = User(url)

        proxy = proxy_apply()
        user_agent = random.choice(user_agent_pool)
        activities = []

        print("start crawl " + url + "\nproxy:" + proxy + '\n')

        #爬取知乎用户动态
        for activity in user.get_activities(proxy, user_agent):
            activities.append(eval(activity))
        user_info['activity'] = activities

        user_info['error'] = ''
        pass_to_writer(user_info)
        proxy_recycle(proxy)
        print(url + 'finished\n')

    except ConnectionError as e:
        print(url + ": " + str(e))
        print("changing the proxy.............\n")
        crawler(user_info)

    except Exception as e:
        user_info['error'] = str(e)
        pass_to_writer(user_info)

        print(url + ": other error happened")
        traceback.print_exc()
<<<<<<< HEAD
        #print(e)
=======
>>>>>>> a6112215125449c29f8fe5990120d23b6ef4272c

#多进程爬虫框架
def multiprocessing_crawler_frame():
    #启动进程池管理进程
    p = multiprocessing.Process(target = proxies_maintain)
    p.start()
    proxy_start_event.wait()

    #启动写进程
    writer_process = multiprocessing.Process(target = writer)
    writer_process.start()

    #获取目标用户id
    source_path = "data/available_users"
    users = get_user_id(source_path)

    start = time.time()

    #使用进程池管理多爬虫进程
    pool = multiprocessing.Pool(number_of_multiprocessing)
    for user in users:
        pool.apply_async(crawler, (user,))

    pool.close()
    pool.join()

    end = time.time()

    #关闭写进程
    write_event.set()
    pass_to_writer(None)
    writer_process.join()

    #关闭代理管理进程
    p.terminate()

    print("finished\ttime cost: " + str(end - start))


if __name__ == "__main__":
    multiprocessing_crawler_frame()
