# -*- codStampToSeries8 -*-

# Requirement
import numpy as np
from scipy import io as spio

#Built-in/Std
import time
import statistics
import os
import os.path

class Stamps:
    month_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def __init__(self, timestamps, begin=1458576000, format = None):
        self.tsl = []
        self.begin = begin
        if not isinstance(timestamps, list):
            raise TypeError("StampToSeries 接受的参数必须是列表类型")
        if format != None:
            for i in range(len(timestamps)):
                timearray = time.strptime(timestamps[i], format)
                self.tsl.append(int(time.mktime(timearray)))
        else:
            for i in range(len(timestamps)):
                self.tsl.append(int(timestamps[i]))

        self.tsl = sorted(self.tsl, reverse = True)

        start = 0
        for i in range(len(self.tsl)):
            if self.tsl[i] <= begin:
                start = i
                break

        self.tsl = self.tsl[start:]

    def contiIntervalCnt(self, interval = 1, begin = None, end = None):
        if self.tsl == []:
            list_len = (begin - end) // 3600
            return [0 for i in range(list_len)]
        else:
            cnt_list = []
            tsl = self.tsl
            if begin == None:
                begin = tsl[0]
            if end == None:
                end = tsl[len(tsl) - 1]
            current = begin
            index = 0
            cnt = 0
            while True:
                if tsl[index] > current:
                    index += 1
                elif tsl[index] > current - interval:
                    cnt += 1
                    index += 1
                else:
                    cnt_list.append(cnt)
                    cnt = 0
                    current = current - interval

                if index == len(tsl):
                    while current > end:
                        cnt_list.append(cnt)
                        cnt = 0
                        current = current - interval
                    break

            return cnt_list

    def fixIntervalCnt(self, inteType):
        if self.tsl == []:
            if inteType == 'day':
                return [0 for i in range(7)]
            if inteType == 'month':
                return [0 for i in range(12)]
            if inteType == 'year':
                return []
        cnt_list = []
        day_cnt = []
        tsl = self.tsl
        month_day = self.month_day
        if inteType == "hour":
            for i in range(24):
                cnt_list.append(0)

            for timestamp in self.tsl:
                timeArray = time.localtime(timestamp)
                cnt_list[timeArray[3]] += 1

        elif inteType == "day":
            begin = time.localtime(tsl[0])[6]
            length = tsl[0] - tsl[len(tsl) - 1]
            base = length // 604800
            redundence = length % 604800
            if redundence % 86400 == 0:
                redundence = redundence // 86400
            else:
                redundence = redundence // 86400 + 1

            for i in range(7):
                cnt_list.append(0)
                day_cnt.append(base)

            for i in range(redundence):
                day_cnt[(begin + i) % 7] += 1

            for timestamp in self.tsl:
                timeArray = time.localtime(timestamp)
                cnt_list[timeArray[6]] += 1

            for i in range(7):
                if day_cnt[i] == 0:
                    day_cnt[i] = 1
                cnt_list[i] = cnt_list[i] / float(day_cnt[i])

        elif inteType == "month":
            begin_time = time.localtime(tsl[0])
            begin_year = begin_time[0]
            begin_month = begin_time[1]
            begin_day = begin_time[2]
            #print(str(begin_year) + "/" + str(begin_month) + "/" + str(begin_day))

            end_time = time.localtime(tsl[len(tsl) - 1])
            end_year = end_time[0]
            end_month = end_time[1]
            end_day = end_time[2]
            #print(str(end_year) + "/" + str(end_month) + "/" + str(end_day))

            for i in range(12):
                cnt_list.append(0)
                day_cnt.append(0)

            if begin_year == end_year and begin_month == end_month:
                day_cnt[begin_month - 1] = begin_day - end_day + 1

            elif begin_year == end_year:
                day_cnt[begin_month - 1] = begin_day
                day_cnt[end_month - 1] = month_day[end_month - 1] - end_day + 1
                if end_month == 2 and begin_year % 4 == 0 and begin_year % 400 != 0:
                    day_cnt[end_month - 1] += 1
                for i in range(begin_month - end_month - 1):
                    day_cnt[begin_month - i - 2] = month_day[begin_month - i - 2]

            else:
                day_cnt[begin_month - 1] = begin_day
                day_cnt[end_month - 1] = month_day[end_month - 1] - end_day + 1
                if end_month == 2 and begin_year % 4 == 0 and begin_year % 400 != 0:
                    day_cnt[end_month - 1] += 1

                for i in range(begin_month - 1):
                    day_cnt[i] += month_day[i]
                    if i == 1 and begin_year % 4 == 0 and begin_year % 400 != 0:
                        day_cnt[i] += 1

                for i in range(end_month, 12):
                    day_cnt[i] += month_day[i]
                    if i == 1 and begin_year % 4 == 0 and begin_year % 400 != 0:
                        day_cnt[i] += 1

                for i in range(begin_year - end_year - 1):
                    year = begin_year - 1 - i
                    for j in range(12):
                        day_cnt[j] += month_day[j]
                        if j == 1 and year % 4 == 0 and year % 400 != 0:
                            day_cnt[j] += 1

            for timestamp in self.tsl:
                timeArray = time.localtime(timestamp)
                cnt_list[timeArray[1] - 1] += 1

            for i in range(12):
                if day_cnt[i] == 0:
                    day_cnt[i] = 1
                cnt_list[i] = cnt_list[i] / float(day_cnt[i])

        elif inteType == "year":
            begin_time = time.localtime(tsl[0])
            begin_year = begin_time[0]
            begin_day = begin_time[7]

            end_time = time.localtime(tsl[len(tsl) - 1])
            end_year = end_time[0]
            end_day = end_time[7]

            year_pass = 2016 - end_year + 1

            for i in range(year_pass):
                cnt_list.append(0)
                day_cnt.append(0)

            if year_pass == 1:
                day_cnt[0] = begin_day - end_day + 1
            elif year_pass == 2:
                day_cnt[0] = begin_day
                if end_year % 4 == 0 and end_year % 400 != 0:
                    day_cnt[1] = 367 - end_day
                else:
                    day_cnt[1] = 366 - end_day
            else:
                day_cnt[0] = begin_day
                if end_year % 4 == 0 and end_year % 400 != 0:
                    day_cnt[year_pass - 1] = 367 - end_day
                else:
                    day_cnt[year_pass - 1] = 366 - end_day
                for i in range(1, year_pass - 1):
                    year = 2016 - i
                    if year % 4 == 0 and year % 400 != 0:
                        day_cnt[i] += 366
                    else:
                        day_cnt[i] += 365

            for timestamp in self.tsl:
                timeArray = time.localtime(timestamp)
                cnt_list[2016 - timeArray[0]] += 1

            for i in range(year_pass):
                if day_cnt[i] == 0:
                    day_cnt[i] = 1
                cnt_list[i] = cnt_list[i] / float(day_cnt[i])
        else:
            raise Exception("interval type unmatch")

        return cnt_list
            #return day_cnt

    # 每固定天数算出一个在小时和weekday上的分布，最终得到一个包含多个分布的序列
    def contiOnFix(self, days, statistic):
        tsl = self.tsl
        begin = self.begin

        if tsl == []:
            return []

        days = days * 86400
        num_statistic = (begin - tsl[len(tsl) - 1]) // days

        final_series = []

        if statistic == 'hour':
            cardinal = 24

            distribution = [0 for i in range(cardinal)]

            k = 1
            lower = begin - k * days

            for i in range(len(tsl)):
                while tsl[i] <= lower:
                    final_series.extend(normalization(distribution))
                    distribution = [0 for i in range(cardinal)]
                    k += 1
                    upper = begin - k * days
                    lower = upper - days

                distribution[time.localtime(tsl[i])[3]] += 1

        elif statistic == 'weekday':
            cardinal = 7

            distribution = [0 for i in range(cardinal)]

            k = 1
            lower = begin - k * days

            for i in range(len(tsl)):
                while tsl[i] <= lower:
                    final_series.extend(normalization(distribution))
                    distribution = [0 for i in range(cardinal)]
                    k += 1
                    upper = begin - k * days
                    lower = upper - days

                distribution[time.localtime(tsl[i])[6]] += 1

        return final_series


def normalization(cnt_list):
    if len(cnt_list) == 0:
        return cnt_list
    elif len(cnt_list) == 1:
        return [0]
    else:
        mean = statistics.mean(cnt_list)
        stdev = statistics.stdev(cnt_list)
        for i in range(len(cnt_list)):
            if stdev == 0:
                cnt_list[i] = 0
            else:
                cnt_list[i] = (cnt_list[i] - mean) / stdev
        return cnt_list

def main():
    #对离散的时间戳做count
    # with open("data/zhihu/users.json", 'r') as input, open("data/zhihu/cnt/hour_series", "w") as output:
    #     for line in input:
    #         line = eval(line)
    #
    #         stamp_list = line["activity"]
    #         stamp = Stamps(stamp_list)
    #         cnt_list = stamp.contiIntervalCnt(begin = 1458576000, end=1250265600, interval = 3600)
    #         # cnt_list = stamp.fixIntervalCnt('hour')
    #         if (len(cnt_list) != 57864):
    #             return 0
    #         out = {}
    #         out['index'] = line['index']
    #         out['count'] = cnt_list
    #         output.write(str(out) + '\n')

    #全局标准化cnt序列
    # rootdir = "data/zhihu/cnt"
    # outputdir = "data/zhihu/norm_cnt"
    # for parent, dirnames, filenames in os.walk(rootdir):
    #     for filename in filenames:
    #         input_path = os.path.join(parent, filename)
    #         output_path = os.path.join(outputdir, filename)
    #         with open(input_path, 'r') as input, open(output_path, 'w') as output:
    #             for line in input:
    #                 line = eval(line)
    #                 line['count'] = normalization(line['count'])
    #                 output.write(str(line) + '\n')

    # 将每连续多天映射到一个相同的分布上
    # options = [[7, 'hour'], [14, 'hour'], [14, 'weekday'], [30, 'hour'], [30, 'weekday']]
    # platforms = ['weibo', 'zhihu']
    #
    # for platform in platforms:
    #     input_path = 'data/{0}/users.json'.format(platform)
    #
    #     for option in options:
    #         print(platform, option)
    #         output_path = 'data/{0}/norm_cnt/{1}_days_{2}'.format(platform, option[0], option[1])
    #
    #         with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
    #             for line in input_file:
    #                 line = eval(line)
    #                 stamp = Stamps(line['activity'])
    #
    #                 out = {}
    #                 out['index'] = line['index']
    #                 out['series'] = stamp.contiOnFix(option[0], option[1])
    #                 output_file.write(str(out) + '\n')

    # 局部标准化hour_series序列
    platforms = ['zhihu', 'weibo']

    for platform in platforms:
        input_path = 'data/{0}/cnt/hour_series'.format(platform)
        out_path = 'data/{0}/norm_cnt/hour_series.txt'.format(platform)

        out_array = np.zeros((1356, 57864))

        with open(input_path, 'r') as cnt:
            for line in cnt:
                line = eval(line)
                index = line['index']
                cnt_list = line['count']
                for i in range(0, len(cnt_list), 24):
                    norm_list = normalization(cnt_list[i:i+24])
                    for j in range(24):
                        out_array[index][i + j] = norm_list[j]

        np.savetxt(out_path, out_array)



if __name__ == "__main__":
    main()
