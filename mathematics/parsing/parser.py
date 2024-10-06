from interface.ibpy import OPERATORS


def prec(c):
    if c == '(':
        return 10
    elif c == '**':
        return 3
    elif c == '/' or c == '*':
        return 2
    elif c == '+' or c == '-':
        return 1
    else:
        return 100  # functions are unary


def associativity(c):
    if c == '**':
        return 'R'
    else:
        return 'L'  # Default to left-associative


def flag_operators(expr):
    """
    It's a bit tricky, because we have to protect operators that contain smaller operators
    sin->_sin_
    asin->_asin_ and not _a_sin__

    Therefore all operators are first substituted with an auxiliary expression, which is substituted back in the end
    flag operators in expression

    protect unary minus signs

    >>> flag_operators("a+b*(c**d-e)**(f+g*h)-i")
    'a_+_b_*_(c_**_d_-_e)_**_(f_+_g_*_h)_-_i'

    :return:
    """

    expr = str(expr)
    operators = OPERATORS
    operators.sort(reverse=True, key=len)
    if expr[0] == '-':
        expr = '(0-1)*' + expr[1:]
    expr = expr.replace('(-', '((0-1)*')
    sub_dict = {}
    for i, op in enumerate(operators):
        expr = expr.replace(op, '_' + '$' + str(i) + '$' + "_")
        sub_dict['$' + str(i) + '$'] = op

    for key, val in sub_dict.items():
        expr = expr.replace(key, val)

    return expr


class ExpressionConverter:
    def __init__(self, infix):
        self.expr = infix
        if not isinstance(self.expr,str):
            self.expr = str(self.expr)

    def postfix(self):
        """
        >>> ExpressionConverter('3465*x**5*sqrt(1 - x**2)/2 - 2205*x**3*sqrt(1 - x**2) + 945*x*sqrt(1 - x**2)/2').postfix()
        '3465,x,5,**,*,1,x,2,**,-,sqrt,*,2,/,2205,x,3,**,*,1,x,2,**,-,sqrt,*,-,945,x,*,1,x,2,**,-,sqrt,*,2,/,+'
        >>> ExpressionConverter("sqrt(1-x**2)/2").postfix()
        '1,x,2,**,-,sqrt,2,/'
        >>> ExpressionConverter('sqrt(385)*(1-9*cos(theta)**2)*sin(3*phi)*sin(theta)**3/(32*sqrt(pi))').postfix()
        '385,sqrt,1,9,theta,cos,2,**,*,-,*,3,phi,*,sin,*,theta,sin,3,**,*,32,pi,sqrt,*,/'
        >>> ExpressionConverter("alpha+b*(c**d-e)**(f+g*h)-i").postfix()
        'alpha,b,c,d,**,e,-,f,g,h,*,+,**,*,+,i,-'
        >>> ExpressionConverter("a+2.2*(c**d-e)**(f+g*h)-i").postfix()
        'a,2.2,c,d,**,e,-,f,g,h,*,+,**,*,+,i,-'
        >>> ExpressionConverter("-3*4").postfix()
        '0,1,-,3,*,4,*'
        >>> ExpressionConverter("sqrt(a*a+b*b)").postfix()
        'a,a,*,b,b,*,+,sqrt'
        >>> ExpressionConverter("sqrt(a*a + b*b)").postfix()
        'a,a,*,b,b,*,+,sqrt'

        :return:
        """
        # remove all white spaces
        self.expr = self.expr.replace(' ','')
        self.expr = flag_operators(self.expr)

        result = []
        stack = []
        op_flag = False
        operand = []
        operator = None

        for i in range(len(self.expr)):
            c = self.expr[i]
            if c == '_':
                if len(operand) > 0:
                    # assemble preceding operand and add it to the result
                    operand = "".join(operand)
                    result.append(operand.strip())
                    operand = []
                if not op_flag:
                    operator = []
                    op_flag = True
                else:
                    # assemble operator expression
                    operator = "".join(operator)
                    # deal with operators
                    while stack and (prec(operator) < prec(stack[-1]) or (
                            prec(operator) == prec(stack[-1]) and associativity(operator) == 'L')) and stack[-1] != '(':
                        result.append(stack.pop())
                    stack.append(operator.strip())
                    op_flag = False

            elif c == '(':
                stack.append(c)
            elif c == ')':
                if len(operand) > 0:
                    result.append("".join(operand))
                    operand = []
                while stack and stack[-1] != '(':
                    result.append(stack.pop())
                stack.pop()  # pop '('
            else:
                if op_flag:
                    operator.append(c)
                else:
                    operand.append(c)

        # pop all the remaining elements from the stack
        while stack:
            if len(operand) > 0:
                result.append("".join(operand))
                operand=""
            result.append(stack.pop())

        return ','.join(result)
