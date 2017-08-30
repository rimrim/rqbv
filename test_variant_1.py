import unittest
from bv import poly_multiply, BV, Rq, modmath, \
    small_samples, large_samples, rot
from timer import Timer
from bitarray import bitarray
from math import log,ceil,floor


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_whole_protocol(self):
        # init parameter for the authentication system
        n = 100
        q = 2**70
        sigma = 3
        t = n
        l = floor(log(t, 2)) - 1
        tau = 20
        bv = BV(n = n, t = t, q = q, sigma = sigma)

        # alice generates her key
        (sk_a, pk_a) = bv.genkey()

        # alice registers her fingerprint

        # alice extracts her iris code
        registered_template = [1 for _ in range(n)]

        # alice encrypts the registered template

        # alice sends the ciphertext to bob

        # bob stores alice template

        # alice authenticates herself

        # alice extract her iris code again

        # alice encrypts the query template

        # alice sends the ciphertext to bob

        # alice prove that she encrypts binary data (ZKP)

        # if the proof pass, bob compute encrypted hd

        # bob mask the encrypted hd

        # bob sends back the ciphertext to alice

        # alice decrypt the hd with her secret key

        # alice sends back the plaintext hd to bob

        # alice prove that she decrypt the hd properly

        # if the proof pass, bob remove the mask and compare HD with threshold

        # Bob output the authentication result


if __name__ == '__main__':
    unittest.main()
