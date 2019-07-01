def parse_file_name(file_name):
    types = ['CINMS', 'DCPP']
    if file_name[0:5] == types[0]:
        first = file_name[0:8]
        second = file_name[9:12]
        third = file_name[13:19]
        forth = file_name[20:26]
    elif file_name[0:4] == types[1]:
        first = file_name[0:7]
        second = file_name[8:11]
        third = file_name[12:18]
        forth = file_name[19:25]
    else:
        raise Exception
    return '%s %s 20%s-%s-%s %s:%s:%s' % (first, second, third[0:2], third[2:4], third[4:6],
                                          forth[0:2], forth[2:4], forth[4:6])


def parse_time2(file_name):
    first, second, third, forth = file_name.split('_')
    return '%s %s 20%s-%s-%s %s:%s:%s' % (first, second, third[0:2], third[2:4], third[4:6],
                                          forth[0:2], forth[2:4], forth[4:6])


if __name__ == '__main__':
    folder = 'CINMS18B_d06_120622_055731.d100.x'
    parse1 = parse_file_name(folder)
    parse2 = parse_time2(folder)
    import datetime
    print(datetime.datetime.strptime(' '.join(parse1.split()[-2:]), '%Y-%m-%d %H:%M:%S'))