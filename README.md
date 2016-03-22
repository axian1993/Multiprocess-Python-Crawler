# Crawler-Python

## 介绍
python实现的知乎和微博的爬虫

zhihu/multiprocessing_crawler_frame.py 为知乎的多进程爬虫框架  
框架支持定时爬取代理，爬虫进程申请、更换、回收代理，随机使用user-agent、多进程同步写文件等功能
>ps.由于免费代理网站提供的代理质量不佳，代码中使用代理部分已被注释


知乎API部分的代码来源于 [egrcc/zhihu-python](https://github.com/egrcc/zhihu-python)，并向egrcc致以感谢
### 修改信息
对User类的parser（）方法进行了修改以支持多代理下的多进程，并增加了get_activities()方法以获取用户动态

## 运行说明
``python3 multiprocessing_crawler_frame.py``

Warning 通常为在使用代理时出现的网络问题，程序会自动为进程更换代理
Error 表明出现了设计框架时为考虑的异常

## 数据存放
用户来源与爬取的数据均存放在 data/ 目录下：
- users_test 存放了用于测试的用户信息
- available_users 用于存放所有需要爬取的用户信息
- users.json 用于存放爬取到的用户动态信息
- error_users.json 用于存放爬取过程中出现Error的用户信息

>ps. 若爬取过程中出现<response 429>, 表明发送请求的速度过快，可适当减少并发进程数，或增加单进程内请求间隔
