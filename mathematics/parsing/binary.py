def to_binary(decimal: int):
    """
    >>> to_binary(8)
    '1000'
    >>> to_binary(15)
    '1111'
    """
    digits = ""
    while decimal>0:
        digit = decimal%2
        decimal-=digit
        decimal//=2
        digits=str(digit)+digits
    return digits

if __name__ == '__main__':
    """
    generate latex file 
    """
    print(r"\hline")
    for i  in range(16,32):
        print(r"&&&&&\\")
        digits = to_binary(i)

        out = str(i)+" &"
        if len(digits)==5:
            pass
        elif len(digits)==4:
            out+=" 0 &"
        elif len(digits)==3:
            out+=" 0 & 0 & "
        elif len(digits)==2:
            out+=" 0 &0  & 0& "
        else:
            out+=" 0&0 & 0& 0& "
        for p in range(len(digits)):
            if p < len(digits)-1:
                out+=digits[p] + " & "
            else:
                out+=digits[p]
        print(out+r"\\")
        print(r"&&&&&\\")
        print(r"\hline ")