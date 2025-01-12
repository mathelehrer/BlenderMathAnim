import re


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def remove_digits(name):
    pattern = r'[0-9]'
    # Match all digits in the string and replace them with an empty string
    return re.sub(pattern, '', name)

    print(new_string)

def remove_punctuation(name):
    pattern = r'[.!?]'
    return re.sub(pattern, '', name)

if __name__ == '__main__':
    print(list(find_all('spam spam spam spam', 'spam')))