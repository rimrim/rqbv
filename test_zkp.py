from unittest import TestCase
import random


def IntegersModP(p):
    class IntegerModP(object):
        def __init__(self, n):
            self.n = n % p
            self.field = IntegerModP

        def __add__(self, other): return IntegerModP(self.n + other.n)

        def __sub__(self, other): return IntegerModP(self.n - other.n)

        def __mul__(self, other): return IntegerModP(self.n * other.n)

        def __truediv__(self, other): return self * other.inverse()

        def __div__(self, other): return self * other.inverse()

        def __neg__(self): return IntegerModP(-self.n)

        def __eq__(self, other): return isinstance(other, IntegerModP) and self.n == other.n

        def __abs__(self): return abs(self.n)

        def __str__(self): return str(self.n)

        def __repr__(self): return '%d (mod %d)' % (self.n, self.p)

        def __divmod__(self, divisor):
            q, r = divmod(self.n, divisor.n)
            return (IntegerModP(q), IntegerModP(r))

        IntegerModP.p = p
        IntegerModP.__name__ = 'Z/%d' % (p)
        return IntegerModP

class Prover(object):
    def __init__(self, g, n, x):
        self.g = g
        self.n = n
        self.x = x

class Verifier(object):
    def __init__(self, g, n):
        self.g = g
        self.n = n

class TestSchnorr(TestCase):
    def setUp(self):
        self.one = 1

    def test_one(self):
        self.assertEqual(0,self.one)

