ascii = {}
ascii_str = ""
for i in range(32,127):
    ascii[i] = chr(i)
    ascii_str += ascii[i]

print(ascii)
print(ascii_str)
print(len(ascii_str))
print(127-32)