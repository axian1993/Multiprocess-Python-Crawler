# -*- coding

#Build-in / Std
import sys, http.cookiejar,

#requirements
import requests, termcolor, html2text
from bs4 import BeautifulSoup

#module
from zhihu import Logging
from zhihu import islogin

request = requests.Session()
request.cookies = http.cookiejar.LWPCookieJar('cookies')

try:
    request.cookies.load('cookies')
except:
    Logging.error("你还没有登陆知乎哦..")
    Logging.info("执行 auth.py 登陆知乎")
    raise exception("无权限（403）")

if islogin != True:
    Logging.error("你的身份信息已经失效，请重新登陆知乎")
    raise exception("无权限（403）")

reload(sys)
sys.setdefaultencoding("utf-8")

class activity(object):
    """docstring for activity"""
    def __init__(self, arg):
        super(activity, self).__init__()
        self.arg = arg
