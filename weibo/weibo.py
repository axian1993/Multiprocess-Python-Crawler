# -*- coding: utf8 -*-
'''
weibo(http://weibo.cn/) crawler by github@qiaoIn
'''

# Build-in / Std
import sys

# requirements
import requests
try:
    from bs4 import BeautifulSoup
except:
    import BeautifulSoup

#reload(sys)
#sys.setdefaultencoding('utf8')

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
                end = user_url.find('?')
                self.user_id = user_url[start:end]
            else:
                start = user_url.find('cn/') + 3
                end = user_url.find('?')
                self.user_name = user_url[start:end]


    def parser(self, user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36'):
        self.requests = requests.Session()
        cookie = {"Cookie": "_T_WM=c2e1ef1af71e2ddad5de1d0eca72a22a; SUHB=010WgxM4nWCT_o; H5_INDEX=3; H5_INDEX_TITLE=bingo%E6%98%AF%E7%8C%AB; SUB=_2A2579jf5DeTxGeVM4lYU-SzMzz2IHXVZGVmxrDV6PUJbrdANLW_jkW1LHetYIeGJu1xoMhi-s3AwsIDVJK9Hcw..; gsid_CTandWM=4uXGdbca1mXK41fzPMzfsdP4A0F; M_WEIBOCN_PARAMS=from%3Dhome"}
        header = {
            'User-Agent': user_agent,
            'Host': 'weibo.cn',
            'Connection': 'keep-alive'
        }

        try:
            r = self.requests.get(self.user_url, headers = header, cookies = cookie, timeout = 30.0)
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
        '''
        TODO 未能在网页中找到关于这个用户名的地方
        '''
        if hasattr(self, 'user_name'):
            return self.user_name
        else:
            if self.soup == None:
                self.parser()
            soup = self.soup
            return


    def get_activities_timestamp(self, user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36'):
        if self.user_url == None:
            print("can't get anonymous user's activities timestamp")
            return
            yield
        else:
            if self.soup == None:
                self.parser()
            soup = self.soup
            pageNum = soup.select('input[type="hidden"]')[0].get('value')

            for page in range(1, int(pageNum)+1):
                #获取 html 页面
                user_url = self.user_url
                equal_index = user_url.find('=')
                user_url_with_page = user_url[:equal_index+1] + str(page)
                cookie = {"Cookie": "_T_WM=c2e1ef1af71e2ddad5de1d0eca72a22a; SUHB=010WgxM4nWCT_o; H5_INDEX=3; H5_INDEX_TITLE=bingo%E6%98%AF%E7%8C%AB; SUB=_2A2579jf5DeTxGeVM4lYU-SzMzz2IHXVZGVmxrDV6PUJbrdANLW_jkW1LHetYIeGJu1xoMhi-s3AwsIDVJK9Hcw..; gsid_CTandWM=4uXGdbca1mXK41fzPMzfsdP4A0F; M_WEIBOCN_PARAMS=from%3Dhome"}
                header = {
                    'User-Agent': user_agent,
                    'Host': 'weibo.cn',
                    'Connection': 'keep-alive'
                }
                r = requests.get(user_url_with_page, headers=header, cookies = cookie)

                soup = BeautifulSoup(r.content, 'html.parser')
                time_list = soup.select('span.ct')

                for timestamp in time_list:
                    time_str = timestamp.get_text()
                    time_str = time_str[:time_str.find('来')]
                    yield time_str


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

