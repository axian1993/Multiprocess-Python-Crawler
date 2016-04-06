# -*- coding: utf8 -*-
'''
weibo(http://weibo.cn/) crawler by github@qiaoIn
'''

# Build-in / Std
import sys,traceback, time

# requirements
import requests
try:
    from bs4 import BeautifulSoup
except:
    import BeautifulSoup

#module
import stamp

#reload(sys)
#sys.setdefaultencoding('utf8')

cookie = {"Cookie": "SUHB=00xoqP1r_i1j5X; _T_WM=c2cde412ee853e53f855434d03bfccde; SUB=_2A256AOiZDeRxGeRJ6VEX8y3Izj6IHXVZCojRrDV6PUJbrdAKLVfXkW1LHeuhuBEY78vecbJGJZQTaf64mbDA6A..; gsid_CTandWM=4ur49e6919MTNaoXfVoRBbqxz7i"}

class User:
    user_url = None
    soup = None
    requests = None


    def __init__(self, user_url):
        if user_url == None:
            raise Exception("这不是一个 user_url")
        else:
            self.user_url = user_url
            index = user_url.find('/u/')
            if index != -1:
                start = index + 3
                # end = user_url.find('?')
                self.user_id = user_url[start:]
            # else:
            #     start = user_url.find('cn/') + 3
            #     end = user_url.find('?')
            #     self.user_name = user_url[start:]


    def parser(self, user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36'):
        self.requests = requests.Session()
        header = {
            'User-Agent': user_agent,
            'Host': 'weibo.cn',
            'Connection': 'keep-alive'
        }

        try:
            while True:
                r = self.requests.get(self.user_url, headers = header, cookies = cookie, timeout = 30.0, allow_redirects = False)
                if r.status_code == 200:
                    break
                time.sleep(3)
        except:
            raise ConnectionError("something wrong with network ...........")
        soup = BeautifulSoup(r.content, 'html.parser')
        self.soup = soup


    def get_user_id(self):
        if hasattr(self, 'user_id'):
            return self.user_id
        else:
            if self.soup == None:
                self.parser()
            soup = self.soup
            attention = soup.select('div > span > a[href^=/attention/]')[0].get('href')
            start = attention.find('?') + 5
            end = int(attention.find('&'))
            self.user_id = attention[start:end]
            return self.user_id


    def get_user_name(self):
        if self.soup == None:
            self.parser()
        soup = self.soup

        ctt_strings = soup.find("span", class_ = "ctt").strings
        user_name = next(ctt_strings)

        return user_name


    def get_activities_timestamp(self, user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36'):
        if self.user_url == None:
            raise Exception("can't get anonymous user's activities timestamp")
        if self.soup == None:
            self.parser()
        soup = self.soup

        pageNum = '1'
        if soup.select('input[type="hidden"]'):
            pageNum = soup.select('input[type="hidden"]')[0].get('value')

        header = {
            'User-Agent': user_agent,
            'Host': 'weibo.cn',
            #'Connection': 'keep-alive'
        }

        page = 1
        while page <= int(pageNum):
            #获取 html 页面
            user_url = self.user_url
            user_url_with_page = user_url + "?page=" + str(page)
            #print(user_url_with_page)
            try:
                r = self.requests.get(user_url_with_page, headers=header, cookies = cookie, allow_redirects = False)
                if r.status_code != 200:
                    time.sleep(1)
                    continue
                soup = BeautifulSoup(r.content, 'html.parser')
                time_list = soup.select('span.ct')

                for timestamp in time_list:
                    time_str = timestamp.get_text()
                    if time_str.find('来自') != -1:
                        time_str = time_str[:time_str.find('来自') - 1]
                    timestamp = stamp.strToStamp(time_str)
                    yield str(timestamp)

                page += 1
            except:
                #traceback.print_exc()
                pass



    def get_user_gender(self, arg):
        pass


    def get_followees_num(self, arg):
        pass


    def get_followers_num(self, arg):
        pass


    def get_post_weibo_num(self, arg):
        pass


    def get_likes_num(self, arg):
        pass


    def get_topics_num(self, arg):
        pass


    def get_followees(self, args):
        pass


    def get_followers(self, arg):
        pass


    def get_post_weibo(self, arg):
        pass


    def get_topics(self, arg):
        pass


class PostWeibo:


    def __init__(self, arg):
        pass


    def get_post_time(self, arg):
        pass


    def get_post_origin(self, arg):
        pass


    def get_repost_num(self, arg):
        pass


    def get_comments_num(self, arg):
        pass


    def get_likes_num(self, arg):
        pass


    def get_post_types(self, arg):
        pass

if __name__ == "__main__":
    user = User("http://weibo.cn/eyesonass")
    timestamps = user.get_activities_timestamp()
    for timestamp in timestamps:
        print(timestamp)
