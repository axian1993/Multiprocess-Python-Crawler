# -*- coding: utf-8 -*-

import time

def strToStamp(timeString):
    current_time = time.time()
    if timeString.find('分钟前') != -1:
        time_pass = timeString[:timeString.find('分钟前')]
        timestamp = int(current_time - int(time_pass) * 60.0)
        timeArray = time.localtime(timestamp)
        form_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        return timestamp
    elif timeString.find('今天') != -1:
        timeArray = time.localtime(current_time)
        form_time = str(timeArray[0]) + "-" + str(timeArray[1]).zfill(2) + "-" + str(timeArray[2]).zfill(2) + " " + timeString[timeString.find("今天") + 3:]
        if len(form_time) == 16:
            form_time = form_time + ":00"
        timeArray = time.strptime(form_time, "%Y-%m-%d %H:%M:%S")
        timestamp = int(time.mktime(timeArray))
        return timestamp
    elif timeString.find('月') != -1:
        timeArray = time.localtime(current_time)
        form_time = str(timeArray[0]) + "-" + timeString[timeString.find('月') - 2:timeString.find('月')].zfill(2) + "-" + timeString[timeString.find('日') - 2:timeString.find('日')].zfill(2) + " " + timeString[timeString.find("日") + 2:]
        if len(form_time) == 16:
            form_time = form_time + ":00"
        timeArray = time.strptime(form_time, "%Y-%m-%d %H:%M:%S")
        timestamp = int(time.mktime(timeArray))
        return timestamp
    else:
        form_time = timeString
        if len(form_time) == 16:
            form_time = form_time + ":00"
        timeArray = time.strptime(form_time, "%Y-%m-%d %H:%M:%S")
        timestamp = int(time.mktime(timeArray))
        return timestamp

if __name__ == "__main__":
    with open('data/raw_users.json', 'r') as input:
        for line in input:
            time_list = eval(line)['activity']
            for timeString in time_list:
                print(strToStamp(timeString))

    print("finished")
