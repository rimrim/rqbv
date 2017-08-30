from unittest import TestCase
import random

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


print('hello')
