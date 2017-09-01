import unittest
import random
from bv import poly_multiply, BV, Rq, modmath, \
    small_samples, large_samples, rot
from timer import Timer
import time
from bitarray import bitarray
from math import log,ceil,floor
from hashlib import sha1 as sha
import numpy

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

        if isinstance(thing, str):
            return len(thing)

        if isinstance(thing, list):
            for i in thing:
                temp = temp + self.size_of(i)
            return temp

        raise ValueError('cannot measure size of this object yet')

class ISISZKP(object):
    def __init__(self, A, y, x, m, n, q):
        # m is lattice dimension
        # n is rows of A
        self.A = A
        self.y = y
        self.x = x

        self.m = m
        self.n = n
        self.q = q

    def schnoor_based_ktx(self, security):
        A = self.A
        y = self.y
        x = self.x
        m = self.m
        n = self.n
        q = self.q
        auth = AuthProtocol()
        print('Stage 1: Commitment')
        print('1.1. sample a random ring element r ...')
        r = Rq.random_samples(self.m, self.q)
        print('1.2. sample a random permutation pi ...')
        pi = list(numpy.random.permutation(self.m))
        pi = Rq(n = m, q = q, coeffs = pi)
        print('1.3. compute 3 commitments ...')
        print('c1 = COMM(pi, A.r mod q) ...')
        c12 = Rq.matrix_mul_ring(A, r)
        bytes_c1 = b''
        for b in c12:
            bytes_c1 += b.encode()
        bytes_c1 = pi.encode() + bytes_c1
        c1 = sha(bytes_c1).hexdigest()
        print('c2 = COMM(pi(r)) ...')
        c2 = Rq.perm_eval(pi, r)
        c2 = Rq(n = m, q = q, coeffs = c2)
        c2 = sha(c2.encode()).hexdigest()
        print('c3 = COMM(pi(x + r)) ...')
        c3 = Rq.perm_eval(pi, x + r)
        c3 = Rq(n = m, q = q, coeffs = c3)
        c3 = sha(c3.encode()).hexdigest()
        comms = [c1, c2, c3]
        print('sending commitments to verifier')
        print('size of commitments is %s'%auth.size_of(comms))
        print('\n')
        print('Stage 2: Challenge, verifier send random ch in {1,2,3} to prover')
        ch = random.choice([1,2,3])
        print('challenge of this round is %s'%ch)
        print('\n')
        print('Stage 3: Response')
        # prover sends response based on challenge
        if ch == 1:
            v = Rq(n = m, q = q, coeffs = Rq.perm_eval(pi, x))
            t = Rq(n = m, q = q, coeffs = Rq.perm_eval(pi, r))
            rsp = [v, t]
            print('ch is 1, prover send back pi(x) and pi(r)')
            comm_size = auth.size_of(rsp)
            print('size of response is %s'%comm_size)
        elif ch == 2:
            phi = pi
            z = x + r
            rsp = [phi, z]
            print('ch is 2, prover send back pi and (x + r)')
            comm_size = auth.size_of(rsp)
            print('size of response is %s'%comm_size)
        else:
            ome = pi
            s = r
            rsp = [ome, s]
            print('ch is 3, prover send back pi and r')
            comm_size = auth.size_of(rsp)
            print('size of response is %s'%comm_size)

        print('\n')
        print('Stage 4: Verification')

        if ch == 1:
            print('check binary...')
            check_c2 = sha(rsp[1].encode()).hexdigest()
            print('check c2')
            if c2 == check_c2:
                print('check pass')
            else:
                print('check fail')
            print('check c3')
            check_c3 = sha((rsp[0]+rsp[1]).encode()).hexdigest()
            if c3 == check_c3:
                print('check pass')
            else:
                print('check fail')

        elif ch == 2:
            az = Rq.matrix_mul_ring(A, z)
            temp = []
            for (i,j) in zip (az, y):
                temp.append(i - j)
            temp_bytes = b''
            for i in temp:
                temp_bytes += i.encode()
            bytes_check_c1 = phi.encode() + temp_bytes
            check_c1 = sha(bytes_check_c1).hexdigest()
            print('check c1')
            if c1 == check_c1:
                print('check pass')
            else:
                print('check fail')
            phi_z = Rq(n = m, q = q, coeffs = Rq.perm_eval(phi,z))
            bytes_check_c3 = phi_z.encode()
            check_c3 = sha(bytes_check_c3).hexdigest()
            print('check c3')
            if c3 == check_c3:
                print('check pass')
            else:
                print('check fail')
        else:
            As = Rq.matrix_mul_ring(A, s)
            As_bytes = b''
            for i in As:
                As_bytes += i.encode()
            bytes_check_c1 = ome.encode() + As_bytes
            check_c1 = sha(bytes_check_c1).hexdigest()
            print('check c1')
            if c1 == check_c1:
                print('check pass')
            else:
                print('check fail')
            ome_s = Rq(n = m, q = q, coeffs = Rq.perm_eval(ome, s))
            bytes_check_c2 = ome_s.encode()
            check_c2 = sha(bytes_check_c2).hexdigest()
            print('check c2')
            if c2 == check_c2:
                print('check pass')
            else:
                print('check fail')




class TestSchnorrProtocol(unittest.TestCase):
    def test_schnoor_based_ktx(self):
        q = 2**20
        n = 3
        m = 100
        a1 = Rq.random_samples(m, q)
        a2 = Rq.random_samples(m, q)
        A = [a1, a2]
        x = Rq(n = m, q = q, coeffs=[1 for _ in range(m)])
        y = Rq.matrix_mul_ring(A,x)

        zkp = ISISZKP(A, y, x, m, n, q)
        zkp.schnoor_based_ktx(60)


    def test_zkp_schnorr_ktx(self):
        # common input is (A,y), prover's witness is x
        # relation is A.x = y mod q, x is binary, and
        # hamming weight wt(x) = w

        # assume that we use some hash function for commitments: sha1
        # from builtin hashlib of python

        # test case A = [a1, a2], or n = 2, m = 5 is the dimension of the
        # lattice (a bit different from other tests where n is the
        # dimension) this is to follow the convention of the paper
        # Ling et al where they use n for something else

        q = 13
        m = 5
        # a1 = Rq.random_samples(m, q)
        # a2 = Rq.random_samples(m, q)
        # print(a1)
        # print(a2)
        # generate test values for A and y
        a1 = Rq(n = m, q = q, coeffs=[0, 2, 5, -5, -4])
        a2 = Rq(n = m, q = q, coeffs=[-6, 0, 1, -3, 3])
        A = [a1, a2]
        y1 = Rq(n = m, q = q, coeffs=[-1, -6, -2, 0, 6])
        y2 = Rq(n = m, q = q, coeffs=[3, -3, -2, 5, 0])
        y = [y1, y2]

        # prover witness x
        x = Rq(n = m, q = q, coeffs=[1, 1, 0, 1, 0])

        # commitments
        # sample a random r is a ring element
        r = Rq.random_samples(m, q)
        # sample a random permutation
        pi = list(numpy.random.permutation(m))
        pi = Rq(n = m, q = q, coeffs = pi)
        # computes 3 commitments
        c12 = Rq.matrix_mul_ring(A, r)
        bytes_c1 = pi.encode() + c12[0].encode() + c12[1].encode()
        c1 = sha(bytes_c1).hexdigest()

        c2 = Rq.perm_eval(pi, r)
        c2 = Rq(n = m, q = q, coeffs = c2)
        c2 = sha(c2.encode()).hexdigest()

        c3 = Rq.perm_eval(pi, x + r)
        c3 = Rq(n = m, q = q, coeffs = c3)
        c3 = sha(c3.encode()).hexdigest()

        comms = [c1, c2, c3]
        # comms is sent to verifier
        auth = AuthProtocol()
        print('size of commitments is %s'%auth.size_of(comms))

        # verifier send a random challenge
        ch = random.choice([1,2,3])
        print('challenge is %s'%ch)

        # prover sends response based on challenge
        if ch == 1:
            v = Rq(n = m, q = q, coeffs = Rq.perm_eval(pi, x))
            t = Rq(n = m, q = q, coeffs = Rq.perm_eval(pi, r))
            rsp = [v, t]
        elif ch == 2:
            phi = pi
            z = x + r
            rsp = [phi, z]
        else:
            ome = pi
            s = r
            rsp = [ome, s]

        print('response size for one round is %s'%auth.size_of(rsp))

        # verifier check rsp
        if ch == 1:
            print('this should be binary %s'%rsp[0])
            check_c2 = sha(rsp[1].encode()).hexdigest()
            if c2 == check_c2:
                print('check pass')
            else:
                print('check fail')
            check_c3 = sha((rsp[0]+rsp[1]).encode()).hexdigest()
            if c3 == check_c3:
                print('check pass')
            else:
                print('check fail')

        elif ch == 2:
            az = Rq.matrix_mul_ring(A, z)
            temp0 = az[0] - y[0]
            temp1 = az[1] - y[1]
            bytes_check_c1 = phi.encode() + temp0.encode() + temp1.encode()
            check_c1 = sha(bytes_check_c1).hexdigest()
            if c1 == check_c1:
                print('check pass')
            else:
                print('check fail')
            phi_z = Rq(n = m, q = q, coeffs = Rq.perm_eval(phi,z))
            bytes_check_c3 = phi_z.encode()
            check_c3 = sha(bytes_check_c3).hexdigest()
            if c3 == check_c3:
                print('check pass')
            else:
                print('check fail')
        else:
            As = Rq.matrix_mul_ring(A, s)
            bytes_check_c1 = ome.encode() + As[0].encode() + As[1].encode()
            check_c1 = sha(bytes_check_c1).hexdigest()
            if c1 == check_c1:
                print('check pass')
            else:
                print('check fail')
            ome_s = Rq(n = m, q = q, coeffs = Rq.perm_eval(ome, s))
            bytes_check_c2 = ome_s.encode()
            check_c2 = sha(bytes_check_c2).hexdigest()
            if c2 == check_c2:
                print('check pass')
            else:
                print('check fail')



    def test_extend_ring_element(self):
        q = 13
        m = 5
        a1 = Rq(n = m, q = q, coeffs=[0, 2, 5, -5, -4])
        a2 = Rq(n = m, q = q, coeffs=[-6, 0, 1, -3, 3])
        a1.extend(a2)
        print(a1)

    def test_list_mult_ele(self):
        q = 13
        m = 5
        a1 = Rq(n = m, q = q, coeffs=[0, 2, 5, -5, -4])
        a2 = Rq(n = m, q = q, coeffs=[-6, 0, 1, -3, 3])
        A = [a1, a2]
        y1 = Rq(n = m, q = q, coeffs=[-1, -6, -2, 0, 6])
        y2 = Rq(n = m, q = q, coeffs=[3, -3, -2, 5, 0])
        y = [y1, y2]

        # prover witness x
        x = Rq(n = m, q = q, coeffs=[1, 1, 0, 1, 0])

        mul = Rq.matrix_mul_ring(A,x)
        self.assertEquals(y, mul)

    def test_encode_ring_element(self):
        x = [5,6,7,8,9]
        n = 5
        q = 7
        x = Rq(n = n, q = q, coeffs=x)
        bytes_x = x.encode()
        hash1 = sha(bytes_x)
        x = [5,6,0,8,16]
        n = 5
        q = 7
        x = Rq(n = n, q = q, coeffs=x)
        bytes_x = x.encode()
        hash2 = sha(bytes_x)
        self.assertEqual(hash1.hexdigest(),hash2.hexdigest())



    # def test_permuation_S_m(self):
        # S_m to be the symmetric group of all permutations of m
        # elements
        # output m polynomials

        # analogy: shuffling a deck of card
        # m = 52
        # we don't need to implement, simply use random.permutation of numpy
        # print(numpy.random.permutation(m))

    def test_evaluate_with_permutation(self):
        # pi = numpy.random.permutation(5)
        # print(list(pi))
        pi = [4, 1, 2, 0, 3]
        x = [5,6,7,8,9]
        x_p = Rq.perm_eval(pi,x)
        # print(pi)
        y = [9, 6, 7, 5, 8]
        self.assertEqual(y,x_p)
        q = 13
        m = 5
        pi = Rq(q = q, n = m, coeffs = pi)
        x = Rq(q = q, n = m, coeffs = x)
        x_p = Rq.perm_eval(pi,x)
        # print(pi)
        y = [9, 6, 7, 5, 8]
        y = Rq(q = q, n = m, coeffs = y)
        self.assertEqual(y,x_p)

    def test_evaluate_with_permutation_rq_element(self):
        pi = [4, 1, 2, 0, 3]
        x = [5,6,7,8,9]
        n = 5
        q = 7
        x = Rq(n = n, q = q, coeffs=x)
        self.assertEqual([2,-1,0,-2,1], Rq.perm_eval(pi,x))

class TestVariant1(unittest.TestCase):

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
        self.assertEqual(size_T, 875)


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
