import unittest
from bv import poly_multiply, BV, Rq, modmath, \
    small_samples, large_samples, rot
from timer import Timer
import time
from bitarray import bitarray
from math import log,ceil,floor

class AuthProtocol(object):
    def __init__(self):
        self.timer = Timer()
        self.steps = {}
        self.start = None
        self.end = None
        self.secs = None
        self.msecs = None

    def start_time_for(self, step):
        if step not in self.steps:
            self.steps[step]={'comp_time':None,'comm_size':None}
        self.start = time.time()

    def stop_time_for(self, step):
        if step not in self.steps:
            raise ValueError('protocol does not have step %s'%s)

        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000

        self.steps[step]['comp_time'] = self.msecs

    def time_for(self, step):
        print('time in msecs for %s is %s'%(step,self.steps[step]['comp_time']))

    def size_of(self, thing):
        temp = 0
        if isinstance(thing, Rq):
            bits = ceil(log(thing.q,2))
            temp = thing.n * bits
            return int(temp / 8)

        if isinstance(thing, list):
            for i in thing:
                temp = temp + self.size_of(i)
            return int(temp / 8)

        raise ValueError('cannot measure size of this object yet')

class TestSchnorrProtocol(unittest.TestCase):
    def test_something(self):
        self.assertEqual(1,0)

class TestVariant1(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_record_time(self):
        auth = AuthProtocol()
        auth.start_time_for("sample")
        a = 2
        b = 200
        c = a**b
        auth.stop_time_for("sample")
        t = auth.time_for("sample")

    def test_size_of(self):
        auth = AuthProtocol()
        n = 100
        q = 2**70
        sigma = 3
        t = n
        T = [1 for _ in range(n)]
        T = Rq(n = n, q = q, coeffs = T)
        size_T = auth.size_of(T)
        self.assertEquals(size_T, 875)


    def test_whole_protocol(self):
        # init parameter for the authentication system
        n = 200
        q = 2**70
        sigma = 3
        t = n
        l = floor(log(t, 2)) - 1
        tau = 20
        bv = BV(n = n, t = t, q = q, sigma = sigma)
        auth = AuthProtocol()

        # alice generates her key
        auth.start_time_for('genkey')
        (sk_a, pk_a) = bv.genkey()
        auth.stop_time_for('genkey')
        # auth.time_for('genkey')

        # alice registers her fingerprint

        # alice extracts her iris code
        T = [1 for _ in range(n)]
        T = Rq(n = n, q = t, coeffs = T)

        # alice encrypts the registered template
        auth.start_time_for('enc_registered_template')
        pack1 = bv.pack1(T)
        enc_T = bv.enc(pack1, pk_a)
        auth.stop_time_for('enc_registered_template')
        auth.time_for('enc_registered_template')

        # alice sends the ciphertext to bob
        print('size of registered template ciphertext %s bytes'%auth.size_of(enc_T))

        # bob stores alice template

        # alice authenticates herself

        # alice extract her iris code again
        Q = [0 for _ in range(n)]
        for i,j in enumerate(Q):
            if i % 2 == 0 or i%3==0 or i% 5==0 or i%7==0 or i%11==0:
                Q[i] = 1

        print('hd of plaintexts queries is %s'%Rq.hd_plain(T,Q))

        # alice encrypts the query template
        pack2 = bv.pack2(Q)
        enc_Q = bv.enc(pack2, pk_a)

        # alice sends the ciphertext to bob
        print('size of query template ciphertext %s bytes'%auth.size_of(enc_Q))

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
