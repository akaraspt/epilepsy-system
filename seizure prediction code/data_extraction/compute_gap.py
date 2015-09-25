import datetime


if __name__ == '__main__':
    format='%Y/%m/%d %H:%M:%S.%f'

    dt1_str = '2003/9/15 08:06:48.0'
    dt1 = datetime.datetime.strptime(dt1_str, format)

    dt2_str = '2003/9/15 08:07:27.0'
    dt2 = datetime.datetime.strptime(dt2_str, format)

    gap_sec = (dt2 - dt1).total_seconds()

    print gap_sec
    print dt1 + datetime.timedelta(seconds=gap_sec)
    print dt2