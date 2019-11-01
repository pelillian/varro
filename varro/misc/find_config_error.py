import sys
import os

def check_config(filename, line_num=None):
    try:
        if line_num:
            os.system("head -{0} {1} > /tmp/find.config".format(int(line_num), filename))
        else:
            os.system("cp {0} /tmp/find.config".format(filename))

        err = os.system("ecppack --svf /dev/null /tmp/find.config /dev/null >/dev/null 2>&1")
    finally:
        os.system("rm /tmp/find.config")
    if err:
        return False
    return True

def search_file(filename):
    if check_config(filename):
        return None
    num_lines = len(open(filename).readlines())
    low = 0
    mid = 0
    high = num_lines
    while (not (high == low == mid)):
        mid = int(((high - low) // 2) + low)
        print(mid)
        if low == mid and not check_config(filename, high):
            break
        if check_config(filename, mid):
            low = mid
        else:
            high = mid
    return mid

def get_broken_lines(filename):
    # NOT IMPLEMENTED
    broken_line = search_file(filename)

    config_not_working = True
    while (config_not_working):
        broken_line = search_file(filename)

if __name__== "__main__":
    if len(sys.argv) != 2:
        print("Incorrect # of arguments.")
        exit(-1)

    broken_line = search_file(sys.argv[1])
    broken_lines = [broken_line]
    if not broken_lines:
        print("ecppack works on file correctly!")
    else:
        for line in broken_lines:
            print(line)

