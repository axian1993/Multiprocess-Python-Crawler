import requests

url = 'http://weibo.cn/digitalboy2'

r = requests.get(url , allow_redirects = False)
print(r.content.decode('utf-8'))
