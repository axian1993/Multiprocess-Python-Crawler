# 修改说明

2016/3/30
## URL处理

- 提供的URL中不包括问号及之后的参数，删去了相应的匹配部分
```python
end = user_url.find('?')
```

- URL后跟的字符串不是用户名，删去了用户名赋值代码
```python
self.user_id = user_url[start:end]
```

## 用户名获取

- 重写了``get_user_name()``方法
```python
def get_user_name(self):
    if self.soup == None:
        self.parser()
    soup = self.soup

    ctt_strings = soup.find("span", class_ = "ctt").strings
    user_name = next(ctt_strings)

    return user_name
```

## 其他

- 修改了get_activities_timestamp()内的一些小问题
