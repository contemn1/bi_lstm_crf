import re

def test():
    pattern = '[\[\]=\n]'
    regex = re.compile(pattern)
    test_string = '[ first/second, third/forth, a=b]'
    print(regex.sub('', test_string))

if __name__ == '__main__':
    test()