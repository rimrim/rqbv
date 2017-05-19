from unittest import TestCase
import random


from modp import IntegersModP

def output_generator(n):
    modn = IntegersModP(n)

    for i in range(2,n):
        l = [0] * n
        for j in range(n):
            ind = modn(i)**j
            ind = int(ind)
            l[ind] = 1
        count = 0
        for k in l:
            if k == 1:
                count=count+1
        if count == n-1:
            return i

class Prover(object):
    def __init__(self, g, n, x):
        self.g = g
        self.n = n
        modn = IntegersModP(n)
        self.x = x

class Verifier(object):
    def __init__(self, g, n):
        self.g = g
        self.n = n

class TestSchnorr(TestCase):
    def setUp(self):
        self.one = 1

    def test_generator(self):
        n = 19
        print(output_generator(n))

    def test_one(self):
        self.assertEqual(0,self.one)
        n = 19



