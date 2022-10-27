from fractions import Fraction
import sys


# Farey sequence
def nearlest_rational(x: float, atol: float):
    int_part = int(x)
    dec_part = x - int_part

    lb = Fraction(0, 1)
    ub = Fraction(1, 1)

    while abs(dec_part - float(lb)) >= atol:
        mid = Fraction(lb.numerator + ub.numerator,
                       lb.denominator + ub.denominator)

        print(f'mid: {mid} = {float(mid)}')

        if dec_part < float(mid):
            ub = mid
        else:
            lb = mid
            
    return [int_part, lb]

if __name__ == '__main__':
    args = sys.argv
    if len(args) >= 2:
        x = float(args[1])
        print(nearlest_rational(x, 1e-5))
    else:
        print(nearlest_rational(3.141517, 1e-5))
