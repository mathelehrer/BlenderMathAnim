from sympy.combinatorics import Permutation
from utils.constants import OPERATORS


def split_rpn(expression):
    """Split an RPN expression into tokens on commas outside quoted literals.

    The comma is what separates the tokens of a formula, so a comma that is
    *part of* a token - the brainfuck instruction ``,``, say - has to be
    protected. Single quotes do that::

        "letter,',',in"    ->  ["letter", "','", "in"]
        "letter,'<',in"    ->  ["letter", "'<'", "in"]

    Quoting is also what tells a literal apart from an operator of the same
    spelling: ``<`` is ``LESS_THAN`` and ``'<'`` is the character. Every
    operator token is unquoted, so an expression written before quoting
    existed splits exactly as it always did.

    :return: the list of tokens, quotes included -
        :func:`~geometry_nodes.nodes.build_function`
        strips them when it reads the literal.
    """
    tokens, token, quoted = [], [], False
    for character in expression:
        if character == "'":
            quoted = not quoted
            token.append(character)
        elif character == "," and not quoted:
            tokens.append("".join(token))
            token = []
        else:
            token.append(character)
    if quoted:
        raise ValueError("unbalanced quote in the expression %r" % expression)
    tokens.append("".join(token))
    return tokens


def operator_priority(c):
    """
    define the priority of an operator
    """
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
    It's a bit tricky because we have to protect operators that contain smaller operators
    sin->_sin_
    asin->_asin_ and not _a_sin__

    Therefore, all operators are first substituted with an auxiliary expression, which is substituted back in the end
    flag operators in expression

    protect unary minus signs

    >>> flag_operators("a+b*(c**d-e)**(f+g*h)-i")
    'a_+_b_*_(c_**_d_-_e)_**_(f_+_g_*_h)_-_i'

    :return:
    """

    expr = str(expr)
    # a sorted copy: sorting OPERATORS itself would reorder the shared list
    operators = sorted(OPERATORS, reverse=True, key=len)
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


def parse_int_tuple(expr):
    # remove parenthesis
    if expr[0] == '(':
        expr = expr[1:]
    if expr[-1] == ')':
        expr = expr[:-1]

    parts = expr.split(',')
    return tuple([int(p) for p in parts])


def parse_permutation(cycle_string):
    perm = Permutation()
    cycles = cycle_string.split(")(")
    for cycle in cycles:
        cycle = cycle.replace(")", "")
        cycle = cycle.replace("(", "")
        numbers = cycle.split(" ")
        numbers = [int(n) for n in numbers]
        if len(numbers) == 1:
            pass
        else:
            perm = perm(*numbers)
    return perm


class ExpressionConverter:
    """One formula, converted between infix and the comma-separated RPN of
    :func:`~geometry_nodes.nodes.make_function`.

    The converter holds the formula in whichever notation it was given;
    :meth:`postfix` reads it as infix and :meth:`infix` reads it as RPN.

    :param expression: the formula, a string or anything whose ``str`` is one
        (a sympy expression, say).
    """

    # RPN token -> infix template, for the scalar operators of make_function.
    # Every template stays within blender's simple-expression subset, so the
    # result runs as a driver without python auto-execution: that is why
    # ** is pow (the simple evaluator has no power operator), % is fmod
    # (which is also what blender's MODULO computes, the sign following the
    # dividend) and the hyperbolic functions are spelled out.
    INFIX_BINARY = {
        "+": "({0}+{1})", "-": "({0}-{1})", "*": "({0}*{1})", "/": "({0}/{1})",
        "**": "pow({0},{1})", "<": "({0}<{1})", ">": "({0}>{1})", "=": "({0}=={1})",
        "%": "fmod({0},{1})", "min": "min({0},{1})", "max": "max({0},{1})",
        "atan2": "atan2({0},{1})",
    }
    INFIX_UNARY = {
        "sin": "sin({0})", "cos": "cos({0})", "tan": "tan({0})",
        "asin": "asin({0})", "acos": "acos({0})", "atan": "atan({0})",
        "sinh": "((exp({0})-exp(-{0}))/2)", "cosh": "((exp({0})+exp(-{0}))/2)",
        "tanh": "((exp(2*{0})-1)/(exp(2*{0})+1))",
        "exp": "exp({0})", "lg": "log({0},10)", "sqrt": "sqrt({0})",
        "abs": "abs({0})", "sgn": "(({0}>0)-({0}<0))", "round": "round({0})",
        "floor": "floor({0})", "ceil": "ceil({0})", "frac": "({0}-floor({0}))",
    }

    def __init__(self, expression):
        self.expr = expression
        if not isinstance(self.expr, str):
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
        # remove all white spaces; a local copy, so the converter still holds
        # the formula it was given
        expr = flag_operators(self.expr.replace(' ', ''))

        result = []
        stack = []
        op_flag = False
        operand = []
        operator = None

        opened_abs = False

        for i in range(len(expr)):
            c = expr[i]
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
                    while stack and (operator_priority(operator) < operator_priority(stack[-1]) or (
                            operator_priority(operator) == operator_priority(stack[-1]) and associativity(operator) == 'L')) and stack[-1] != '(':
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
            elif c == '|' and not opened_abs:
                stack.append(c)
                opened_abs = True
            elif c == '|' and opened_abs:
                opened_abs = False
                if len(operand) > 0:
                    result.append("".join(operand))
                    operand = []
                while stack and stack[-1] != '|':
                    result.append(stack.pop())
                result.append('abs')
                stack.pop()  # pop '|'
            else:
                if op_flag:
                    operator.append(c)
                else:
                    operand.append(c)

        # pop all the remaining elements from the stack
        while stack:
            if len(operand) > 0:
                result.append("".join(operand))
                operand = ""
            result.append(stack.pop())

        return ','.join(result)

    def infix(self, variables={}):
        """The formula, read as RPN, written out in infix.

        The inverse of :meth:`postfix`, fully parenthesised so that no
        precedence rule is needed to read it back. The output only uses
        blender's simple-expression subset, so it can go straight into a
        driver (:func:`~interface.ibpy.attach_driver`) without python
        auto-execution.

        >>> ExpressionConverter("t,2,**,1,-").infix()
        '(pow(t,2)-1)'
        >>> ExpressionConverter("pi,t,*,cos").infix({"t": "frame/60"})
        'cos((pi*(frame/60)))'
        >>> ExpressionConverter("t,3,min,lg").infix()
        'log(min(t,3),10)'
        >>> ExpressionConverter("a,b,%,sgn").infix()
        '((fmod(a,b)>0)-(fmod(a,b)<0))'
        >>> ExpressionConverter(ExpressionConverter("sqrt(1-x**2)/2").postfix()).infix()
        '(sqrt((1-pow(x,2)))/2)'
        >>> ExpressionConverter("a,b,add").infix()
        Traceback (most recent call last):
        ...
        ValueError: 'add' in 'a,b,add' has no scalar infix form

        :param variables: ``{name: infix expression}``, substituted (in
            parentheses) for every occurrence of the token ``name``. Names that
            are not listed are written as they are - the variables of a driver,
            or a constant such as ``pi``.
        :raises ValueError: on an operator without a scalar infix form (the
            vector operators of make_function), a quoted literal, a token that
            is neither a name nor a number, or a formula that does not reduce
            to exactly one value.
        """
        stack = []
        for token in split_rpn(self.expr):
            token = token.strip()
            if token == "":
                continue
            if token in variables:
                stack.append("(%s)" % variables[token])
            elif token in self.INFIX_BINARY:
                if len(stack) < 2:
                    raise ValueError("%r in %r has no two operands" % (token, self.expr))
                right, left = stack.pop(), stack.pop()
                stack.append(self.INFIX_BINARY[token].format(left, right))
            elif token in self.INFIX_UNARY:
                if not stack:
                    raise ValueError("%r in %r has no operand" % (token, self.expr))
                stack.append(self.INFIX_UNARY[token].format(stack.pop()))
            elif token in OPERATORS:
                raise ValueError("%r in %r has no scalar infix form" % (token, self.expr))
            elif token.isidentifier():
                stack.append(token)
            else:
                try:
                    float(token)
                except ValueError:
                    raise ValueError("%r in %r is neither an operator, a name nor a number"
                                     % (token, self.expr))
                stack.append(token)
        if len(stack) != 1:
            raise ValueError("%r leaves %d values on the stack, not one" % (self.expr, len(stack)))
        return stack[0]
