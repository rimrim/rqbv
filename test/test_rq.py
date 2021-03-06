from __future__ import absolute_import
import unittest

import cProfile
from math import floor
from math import log
from unittest import TestCase
from bitstring import BitArray

import math

import random

from auth_protocol import AuthProtocol, Substractor, GarbleCircuit, Comparator, dec, enc, GateAnd, GateOr, GateXNOr
from bv import poly_multiply, BV, BGV, small_samples, large_samples

from ntt import *
from mymath import *
from timer import Timer
from os import path, remove

import logging, autologging

class Test_GC(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_print_truth_table(self):
    #     for j in (0,1):
    #         for l in (0,1):
    #             for m in (0,1):
    #                 print(j, l, m)

    def test_aes(self):
        gc = GarbleCircuit()
        k = b'b'*16
        m = b'a'*16
        c = enc(k, m)
        p = dec(k, c)
        self.assertEqual(p,m)

    def test_garble_a_gate_substractor(self):
        gc = GarbleCircuit()
        subs = Substractor()
        subs.assign_random_keys()
        gc.garble(subs)
        self.assertEqual(len(subs.garbled_keys_in[0][0]),16)

        k0 = subs.garbled_keys_in[0][0]
        k1 = subs.garbled_keys_in[1][0]
        k2 = subs.garbled_keys_in[2][0]
        out0, out1 = subs.eval_keys(k0, k1, k2)

        self.assertEqual(out0, subs.garbled_keys_out[0][0])
        self.assertEqual(out1, subs.garbled_keys_out[1][0])


        k0 = subs.garbled_keys_in[0][0]
        k1 = subs.garbled_keys_in[1][0]
        k2 = subs.garbled_keys_in[2][1]
        out0, out1 = subs.eval_keys(k0, k1, k2)
        self.assertEqual(out0, subs.garbled_keys_out[0][1])
        self.assertEqual(out1, subs.garbled_keys_out[1][1])

        k0 = subs.garbled_keys_in[0][0]
        k1 = subs.garbled_keys_in[1][1]
        k2 = subs.garbled_keys_in[2][0]
        out0, out1 = subs.eval_keys(k0, k1, k2)
        self.assertEqual(out0, subs.garbled_keys_out[0][1])
        self.assertEqual(out1, subs.garbled_keys_out[1][1])


    def test_garble_a_gate_comparator(self):
        gc = GarbleCircuit()
        comp = Comparator()
        comp.assign_random_keys()
        gc.garble(comp)
        self.assertEqual(len(comp.garbled_keys_in[0][0]),16)

        k0 = comp.garbled_keys_in[0][0]
        k1 = comp.garbled_keys_in[1][0]
        out = comp.eval_keys(k0, k1)
        self.assertEqual(out, comp.garbled_keys_out[0][0])

        k0 = comp.garbled_keys_in[0][0]
        k1 = comp.garbled_keys_in[1][1]
        out = comp.eval_keys(k0, k1)
        self.assertEqual(out, comp.garbled_keys_out[0][0])

        k0 = comp.garbled_keys_in[0][1]
        k1 = comp.garbled_keys_in[1][0]
        out = comp.eval_keys(k0, k1)
        self.assertEqual(out, comp.garbled_keys_out[0][1])

        k0 = comp.garbled_keys_in[0][1]
        k1 = comp.garbled_keys_in[1][1]
        out = comp.eval_keys(k0, k1)
        self.assertEqual(out, comp.garbled_keys_out[0][0])

    def test_garble_a_gate_and(self):
        gc = GarbleCircuit()
        comp = GateAnd()
        comp.assign_random_keys()
        gc.garble(comp)
        self.assertEqual(len(comp.garbled_keys_in[0][0]),16)

        k0 = comp.garbled_keys_in[0][0]
        k1 = comp.garbled_keys_in[1][0]
        out = comp.eval_keys(k0, k1)
        self.assertEqual(out, comp.garbled_keys_out[0][0])

        k0 = comp.garbled_keys_in[0][0]
        k1 = comp.garbled_keys_in[1][1]
        out = comp.eval_keys(k0, k1)
        self.assertEqual(out, comp.garbled_keys_out[0][0])

        k0 = comp.garbled_keys_in[0][1]
        k1 = comp.garbled_keys_in[1][0]
        out = comp.eval_keys(k0, k1)
        self.assertEqual(out, comp.garbled_keys_out[0][0])

        k0 = comp.garbled_keys_in[0][1]
        k1 = comp.garbled_keys_in[1][1]
        out = comp.eval_keys(k0, k1)
        self.assertEqual(out, comp.garbled_keys_out[0][1])

    def test_xnor(self):
        a = GateXNOr()
        a.input[0] = 1
        a.input[1] = 1
        a.eval()
        self.assertEqual(a.output[0],1)
        a.input[0] = 0
        a.input[1] = 1
        a.eval()
        self.assertEqual(a.output[0],0)
        a.input[0] = 0
        a.input[1] = 0
        a.eval()
        self.assertEqual(a.output[0],1)
        a.input[0] = 1
        a.input[1] = 0
        a.eval()
        self.assertEqual(a.output[0],0)

    def test_or(self):
        a = GateOr()
        a.input[0] = 1
        a.input[1] = 1
        a.eval()
        self.assertEqual(a.output[0],1)
        a.input[0] = 0
        a.input[1] = 1
        a.eval()
        self.assertEqual(a.output[0],1)
        a.input[0] = 0
        a.input[1] = 0
        a.eval()
        self.assertEqual(a.output[0],0)
        a.input[0] = 1
        a.input[1] = 0
        a.eval()
        self.assertEqual(a.output[0],1)

    def test_and_gate(self):
        a = GateAnd()
        a.input[0] = 1
        a.input[1] = 1
        a.eval()
        self.assertEqual(a.output[0],1)
        a.input[0] = 0
        a.input[1] = 1
        a.eval()
        self.assertEqual(a.output[0],0)
        a.input[0] = 0
        a.input[1] = 0
        a.eval()
        self.assertEqual(a.output[0],0)
        a.input[0] = 1
        a.input[1] = 0
        a.eval()
        self.assertEqual(a.output[0],0)

    def test_substractor(self):
        s = Substractor()
        s.input[0] = 0
        s.input[1] = 1
        s.input[2] = 0
        s.eval()
        self.assertEqual(s.output[0],1)
        self.assertEqual(s.output[1],1)

    def test_generate_circuit(self):
        auth = AuthProtocol()
        # check whether a - b > tau or not, in plaintext
        hd = BitArray('0b1011')
        r = BitArray('0b0101')
        tau = BitArray('0b1001')
        output = auth.generate_circuit(hd, r, tau)
        self.assertEqual(output, 0)

        tau = BitArray('0b0011')
        output = auth.generate_circuit(hd, r, tau)
        self.assertEqual(output, 1)

    def test_generate_garble_circuit(self):
        auth = AuthProtocol()
        gc = GarbleCircuit()
        l = 4
        gc.generate_random_keys_for_circuit(l)
        for i in range(l):
            gc.comparators[i].garbled_keys_in[1] = (b'b'*16,b'b'*16)

        gc.generate_gc(l)
        for i in range(l):
            self.assertEqual(gc.comparators[i].garbled_keys_in[0],gc.substractors[i].garbled_keys_out[0])
            if i != (l-1):
                self.assertEqual(gc.comparators[i].garbled_keys_out[0],gc.ors[i].garbled_keys_in[1])
            else:
                self.assertEqual(gc.comparators[i].garbled_keys_out[0],gc.ands[i-1].garbled_keys_in[0])
            if i < l -2:
                self.assertEqual(gc.ors[i].garbled_keys_in[0], gc.ands[i].garbled_keys_out[0])
                self.assertEqual(gc.xnors[i].garbled_keys_in[0], gc.substractors[i].garbled_keys_out[0])
                self.assertEqual(gc.xnors[i].garbled_keys_out[0], gc.ands[i].garbled_keys_in[1])
            if i < l-1:
                self.assertEqual(gc.substractors[i].garbled_keys_in[2], gc.substractors[i+1].garbled_keys_out[1])
                self.assertEqual(gc.comparators[i].garbled_keys_in[1],(b'b'*16,b'b'*16))
                self.assertEqual(gc.xnors[i].garbled_keys_in[1], gc.comparators[i].garbled_keys_in[1])

    def test_evaluate_garble_circuit_false(self):
        # assume a circuit was generated
        auth = AuthProtocol()
        gc = GarbleCircuit()
        l = 4
        gc.generate_random_keys_for_circuit(l)
        gc.generate_gc(l)

        # simulate a client with correct keys
        hd = BitArray('0b1011')
        r = BitArray('0b0101')
        tau = BitArray('0b1001')
        k_hd = ['' for _ in range(l)]
        k_r = ['' for _ in range(l)]
        k_tau = ['' for _ in range(l)]
        for i in range(l):
            k_hd[i] = gc.substractors[i].garbled_keys_in[0][int(hd[i])]
            k_r[i] = gc.substractors[i].garbled_keys_in[1][int(r[i])]
            k_tau[i] = gc.comparators[i].garbled_keys_in[1][int(tau[i])]

        key_out = gc.eval_garbled_circuit(k_hd, k_r, k_tau)
        self.assertEqual(key_out, gc.ors[0].garbled_keys_out[0][0])

    def test_evaluate_garble_circuit_true(self):
        # assume a circuit was generated
        auth = AuthProtocol()
        gc = GarbleCircuit()
        l = 4
        gc.generate_random_keys_for_circuit(l)
        gc.generate_gc(l)

        # simulate a client with correct keys
        hd = BitArray('0b1011')
        r = BitArray('0b0101')
        tau = BitArray('0b0001')
        k_hd = ['' for _ in range(l)]
        k_r = ['' for _ in range(l)]
        k_tau = ['' for _ in range(l)]
        for i in range(l):
            k_hd[i] = gc.substractors[i].garbled_keys_in[0][int(hd[i])]
            k_r[i] = gc.substractors[i].garbled_keys_in[1][int(r[i])]
            k_tau[i] = gc.comparators[i].garbled_keys_in[1][int(tau[i])]

        key_out = gc.eval_garbled_circuit(k_hd, k_r, k_tau)
        self.assertEqual(key_out, gc.ors[0].garbled_keys_out[0][1])

@autologging.logged(logging.getLogger('homocrypto'))
class TestOperationOnPlainText(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_HD(self):
        n = 16
        x = [1 for _ in range(n)]
        y = [0 for _ in range(n)]
        y[0] = 1
        y[1] = 1
        y[n - 1] = 1
        y[n - 2] = 1
        y[n//2] = 1

        def rot(l, n):
            return l[n:] + l[:n]
        def add(l1, l2):
            return [x + y for x,y in zip(l1, l2)]

        def hamming_weight(z):
            # lay nua sau
            n  = len(z)
            while(n != 1):
                n = math.ceil(n/2)
                z1 = z[:n]
                z2 = z[n:]
                if(len(z2) < len(z1)):
                    z2.append(0)
                z = add(z1, z2)
            return z[0]

        def hamming_weight2(c):
            for i in range(math.floor(math.log(n,2))):
                temp = rot(c, 2**i)
                c = add(c, temp)
            return c[0]

        z = []
        for i,j in zip(x,y):
            t = (i + j) % 2
            z.append(t)
        print(x)
        print(y)
        print(hamming_weight2(z))

        def hamming_distance(x,y):
            z = []
            for i,j in zip(x,y):
                t = (i + j) % 2
                z.append(t)
            return hamming_weight(z)
        self.assertEqual(hamming_distance(x,y), 11)


class TestNTT(TestCase):

    def test_ntt_bar(self):
        n = 8
        qbits = 5
        q = set_up_params(n, qbits)
        print('n = %s'%n)
        print('q = %s'%q)
        k = (q-1)//n
        print('k = %s'%k)
        g = find_generator(q-1, q)
        omega = modmath(g**k, q)
        print('omega = %s'%omega)
        psi = square_root_mod(q, n)
        print('psi = %s'%psi)
        a = [1,1,1,4,5,6,7,8]
        a = Rq(n,q,a)

        x_j = 1
        y_j = 0
        a0 = (a.evaluate_at(psi**((3**(0 + x_j))*((n-1)**(0+y_j)))))
        a1 = (a.evaluate_at(psi**((3**(1 + x_j))*((n-1)**(0+y_j)))))
        a2 = (a.evaluate_at(psi**((3**(2 + x_j))*((n-1)**(0+y_j)))))
        a3 = (a.evaluate_at(psi**((3**(3 + x_j))*((n-1)**(0+y_j)))))
        a4 = (a.evaluate_at(psi**((3**(0 + x_j))*((n-1)**(1+y_j)))))
        a5 = (a.evaluate_at(psi**((3**(1 + x_j))*((n-1)**(1+y_j)))))
        a6 = (a.evaluate_at(psi**((3**(2 + x_j))*((n-1)**(1+y_j)))))
        a7 = (a.evaluate_at(psi**((3**(3 + x_j))*((n-1)**(1+y_j)))))
        abar = [a0,a1,a2,a3,a4,a5,a6,a7]
        abar = Rq(n, q, abar)

        abar2 = transform_plus_bar(a, omega, psi, q)
        self.assertEqual(abar, abar2)


    def test_new_ntt_order(self):
        n = 8
        qbits = 3
        q = set_up_params(n, qbits)
        # print('n = %s'%n)
        # print('q = %s'%q)
        k = (q-1)//n
        # print('k = %s'%k)
        g = find_generator(q-1, q)
        omega = modmath(g**k, q)
        # print('omega = %s'%omega)
        psi = square_root_mod(q, n)
        # print('psi = %s'%psi)
        a = [1,1,1,4,5,6,7,8]
        a = Rq(n,q,a)
        ahat = transform_plus(a, omega, psi, q)
        # print(ahat)
        x_j = 1
        y_j = 0

        a0 = (a.evaluate_at(psi**((3**(0 + x_j))*((n-1)**(0+y_j)))))
        a1 = (a.evaluate_at(psi**((3**(1 + x_j))*((n-1)**(0+y_j)))))
        a2 = (a.evaluate_at(psi**((3**(2 + x_j))*((n-1)**(0+y_j)))))
        a3 = (a.evaluate_at(psi**((3**(3 + x_j))*((n-1)**(0+y_j)))))
        a4 = (a.evaluate_at(psi**((3**(0 + x_j))*((n-1)**(1+y_j)))))
        a5 = (a.evaluate_at(psi**((3**(1 + x_j))*((n-1)**(1+y_j)))))
        a6 = (a.evaluate_at(psi**((3**(2 + x_j))*((n-1)**(1+y_j)))))
        a7 = (a.evaluate_at(psi**((3**(3 + x_j))*((n-1)**(1+y_j)))))
        abar = [a0,a1,a2,a3,a4,a5,a6,a7]
        abar = Rq(n, q, abar)
        print(abar)

        x_j = 2
        y_j = 0
        a0 = (a.evaluate_at(psi**((3**(0 + x_j))*((n-1)**(0+y_j)))))
        a1 = (a.evaluate_at(psi**((3**(1 + x_j))*((n-1)**(0+y_j)))))
        a2 = (a.evaluate_at(psi**((3**(2 + x_j))*((n-1)**(0+y_j)))))
        a3 = (a.evaluate_at(psi**((3**(3 + x_j))*((n-1)**(0+y_j)))))
        a4 = (a.evaluate_at(psi**((3**(0 + x_j))*((n-1)**(1+y_j)))))
        a5 = (a.evaluate_at(psi**((3**(1 + x_j))*((n-1)**(1+y_j)))))
        a6 = (a.evaluate_at(psi**((3**(2 + x_j))*((n-1)**(1+y_j)))))
        a7 = (a.evaluate_at(psi**((3**(3 + x_j))*((n-1)**(1+y_j)))))
        rot1 = [a0,a1,a2,a3,a4,a5,a6,a7]
        rot1 = Rq(n, q, rot1)
        print(rot1)

    def test_ntt_and_ntt_plus_evaluate_at(self):
        n = 8
        qbits = 5
        q = set_up_params(n, qbits)
        print('n = %s'%n)
        print('q = %s'%q)
        k = (q-1)//n
        print('k = %s'%k)
        g = find_generator(q-1, q)
        omega = modmath(g**k, q)
        print('omega = %s'%omega)
        psi = square_root_mod(q, n)
        print('psi = %s'%psi)
        in1 = [1,2,3,4,5,6,7,8]
        in1 = Rq(n, q, in1)
        nttin1 = transform(in1,omega,q)
        self.assertEqual(inverse_transform(nttin1, omega, q),in1)
        self.assertEqual(in1.evaluate_at(1),nttin1[0])
        self.assertEqual(in1.evaluate_at(omega),nttin1[1])
        self.assertEqual(in1.evaluate_at(omega**2),nttin1[2])
        self.assertEqual(in1.evaluate_at(omega**3),nttin1[3])
        self.assertEqual(in1.evaluate_at(omega**4),nttin1[4])
        self.assertEqual(in1.evaluate_at(omega**5),nttin1[5])
        self.assertEqual(in1.evaluate_at(omega**6),nttin1[6])
        self.assertEqual(in1.evaluate_at(omega**7),nttin1[7])

        ntt_plus_1 = transform_plus(in1,omega,psi,q)
        inv_ntt_plus_1 = inverse_transform_plus(ntt_plus_1, omega, psi, q)
        self.assertEqual(inv_ntt_plus_1, in1)
        self.assertEqual(in1.evaluate_at(psi), ntt_plus_1[0])
        self.assertEqual(in1.evaluate_at(psi**3), ntt_plus_1[1])
        self.assertEqual(in1.evaluate_at(psi**5), ntt_plus_1[2])
        self.assertEqual(in1.evaluate_at(psi**7), ntt_plus_1[3])
        self.assertEqual(in1.evaluate_at(psi**9), ntt_plus_1[4])
        self.assertEqual(in1.evaluate_at(psi**11), ntt_plus_1[5])
        self.assertEqual(in1.evaluate_at(psi**13), ntt_plus_1[6])
        self.assertEqual(in1.evaluate_at(psi**15), ntt_plus_1[7])
        #
        # in2 = [-1, -4, -10, 19]
        # print(inverse_transform_plus(in2, omega, psi, q))


    def test_ntt(self):
        n = 5
        t = 11
        g = 4
        m1 = Rq(n, t, [1,0,0,1,1])
        m2 = Rq(n, t, [1,0,1,1,1])
        # print(m1*m2)

        m1crt = inverse_transform(m1, g, t)
        m2crt = inverse_transform(m2, g, t)
        # print(m1crt)
        # print(m2crt)
        pro = Rq(n, t, [3, 1, 6, 6, 1])
        # m1crt = Rq(n, t, m1crt)
        # m2crt = Rq(n, t, m2crt)
        # pro = m1crt*m2crt
        pro = Rq.positive_q(pro, t)
        res = (transform(pro, g, t))
        # print(res)



    def test_speed(self):
        n = 2**10
        invec = [random.randint(0,1) for _ in range(n)]
        omega = 10302
        q = 12289
        outvec = []
        with Timer() as t:
            outvec = transform(invec, omega, q)
        print("time to do ntt %s"%t.msecs)
    def test_forward(self):
        n = 5
        q = 11
        invec = [1,0,0,1,1]
        outvec = transform(invec, 4, 11)
        self.assertEqual([3,2,3,-2,-1], outvec)

    def test_inverse(self):
        n = 5
        q = 11
        invec = [3,2,3,9,10]
        outvec = inverse_transform(invec, 4, 11)
        self.assertEqual(outvec, [1,0,0,1,1])

    def test_circular_convolve(self):
        in1 = [1,0,0,1,1]
        in2 = [1,1,1,1,1]
        p = circular_convolve(in1,in2)
        pass

    def test_mult_componentwise(self):
        n = 10
        t = 11
        g = 2
        in1 = [1,1,0,1,1,0,0,1,0,0]
        in2 = [1,0,0,0,1,0,0,1,0,0]
        inv1 = inverse_transform(in1,g, t)
        inv2 = inverse_transform(in2,g, t)

        cir = circular_convolve(inv1,inv2)
        cirmod = [i%t for i in cir]

        compwise = transform(cirmod, g, 11)

        self.assertEqual(compwise,[1,0,0,0,1,0,0,1,0,0])

    def test_XOR(self):
        # t XOR q = t + q - 2tq, component wise, not circular convolution
        n = 10
        t = 11
        g = 2
        in1 = [1,1,0,1,1,0,0,1,0,0]
        in2 = [1,0,0,0,1,0,0,1,0,0]
        add = [i + j for i,j in zip(in1, in2)]

        # first way, cannot be applied with ciphertext, noise will be large
        mul2 = [-2*i*j for i,j in zip(in1, in2)]
        xor1 = [i + j for i,j in zip(add, mul2)]

        #second way, can be applied in cihpertext, noise will be small
        inv1 = inverse_transform(in1,g, t)
        inv2 = inverse_transform(in2,g, t)

        # mul homomorphically
        cir = circular_convolve(inv1,inv2)
        cirmod = [i%t for i in cir]

        compwise = transform(cirmod, g, 11)
        # mul with const -2
        mul2 = [-2*i for i in compwise]
        xor2 = [i + j for i,j in zip(add, mul2)]
        self.assertEqual(xor1,xor2)

    def test_circular_conv(self):
        n = 5
        t = 11
        g = 4
        in1 = [1,0,0,1,1]
        out1 = transform(in1, g, t)
        inv = inverse_transform(out1, 4, t)
        in2 = [1,1,1,0,1]
        out2 = transform(in2, g, t)
        compwise = [(i*j)%t for i,j in zip(out1,out2)]
        a = (inverse_transform(compwise,g,t))
        b = (circular_convolve(in1,in2))
        # print(b)
        self.assertEqual(a,b)

    def test_scale_change_mod(self):
        t = 17
        g = 9
        psi = 3
        n = 8
        in1 = [1,0,0,1,1,0,1,1]
        in2 = [1,0,0,0,1,0,0,1]
        in1 = Rq(n, t, in1)
        in2 = Rq(n, t, in2)

        nttp = transform_plus(in1, g, psi, t)
        in3 = [-1, -1, -2, 4, 8, -5, 0, 5]
        inntp = inverse_transform_plus(in3, g, psi, t)

        convp = convolution_plus(in1,in2,g,psi,t)
        self.assertEqual(convp, in1*in2)


    def test_ntt_the_other_root(self):
        t = 17
        g = 9
        psi = 3
        n = 8
        in1 = [1,0,0,1,1,0,1,1]
        # out1 = find_params_and_transform(in1, t)
        out1 = transform(in1, g, t)
        in2 = [1,0,0,0,0,0,0,1]
        out2 = transform(in2, g, t)
        conv = circular_convolve(in1, in2)

        # first way of computing product mod x^n + 1, have to pad with 0s
        in1 = [1,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0]
        in2 = [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        conv = circular_convolve(in1, in2)
        conv = Rq(8, t, conv)

        # second way of computing mod x^n + 1
        sin1 = []
        sin2 = []
        for i in range(n):
            sin1.append((psi**i)*in1[i])
            sin2.append((psi**i)*in2[i])
        sin1 = Rq(n, t, sin1)
        sin2 = Rq(n, t, sin2)
        convsin = circular_convolve(sin1,sin2)
        psiinv = reciprocal(psi, t)
        invs = []
        for i in range(n):
            invs.append(psiinv**i)
        invs = Rq(n, t, invs)
        prod = [i*j for i,j in zip(convsin,invs)]
        prod = Rq(n, t, prod)
        self.assertEqual(prod, conv)

        in1 = [1,0,0,1,1,0,1,1]
        in2 = [1,0,0,0,0,0,0,1]
        # print(ntt_ring_mult(in1,in2,g,t))





class TestBGV(TestCase):
    def test_key_gen(self):
        n = 5
        q = 2**20
        # generator
        # t need to be a prime to find n roots of unity
        g = 4
        t = 11
        alpha_q = 3

        bgv = BGV(n, q, t, alpha_q)

        sk = bgv.sec_key_gen()
        s = sk[1]
        pk = bgv.pub_key_gen(s)
        m = Rq(n, t, [1,0,0,1,1])
        c = bgv.encrypt(m, pk)
        # print(bgv.check_noise(c,sk))
        p = bgv.decrypt(c, sk)
        self.assertEqual(p,[1,0,0,1,1])

        m2 = Rq(n, t, [1,1,1,0,0])
        c2 = bgv.encrypt(m2, pk)
        p2 = bgv.decrypt(c2, sk)
        # addition homomorphically
        c_add = bgv.add(c,c2)
        p_add = bgv.decrypt(c_add, sk)
        self.assertEqual(p_add, [2,1,1,1,1])
        # substraction homomorphically
        c_sub = bgv.subs(c,c2)
        p_sub = bgv.decrypt(c_sub, sk)
        self.assertEqual(p_sub, [0,-1,-1,1,1])


        # multiply in coefficient domain, circular convolution
        c_mul = bgv.mul(c,c2)
        p_mul = bgv.decrypt(c_mul, sk)
        m1m2_coeff = m*m2
        self.assertEqual(p_mul, m1m2_coeff)

    def test_mul_comp(self):
        n = 5
        q = 2**20
        # generator
        # t need to be a prime to find n roots of unity
        g = 4
        t = 11
        alpha_q = 3

        bgv = BGV(n, q, t, alpha_q)

        sk = bgv.sec_key_gen()
        s = sk[1]
        pk = bgv.pub_key_gen(s)

        # multiply component wise, using NTT
        m1 = Rq(n, t, [1,1,0,1,1])
        m2 = Rq(n, t, [1,0,0,0,1])
        # first need to encode them interms of iNTT
        # try in the plaintext first
        im1 = inverse_transform(m1,g,t)
        # print(im1)
        # im1 = Rq(n, t, im1)
        im2 = inverse_transform(m2,g,t)
        # im2 = Rq(n, t, im2)
        # print(im2)

        polymul = ntt_poly_mult(im1, im2, g, t)
        # print('ntt poly mult %s'%polymul)

        cir = circular_convolve(im1,im2)
        # print('x^n - 1 poly mult im1 im2 %s'%cir)
        cirmod = Rq(n, t, cir)
        # print('x^n - 1 poly mult mod t %s'%cirmod)
        compwise = transform(cirmod, g, t)
        self.assertEqual(compwise,[1,0,0,0,1])

        # now the ciphertexts
        cim1 = bgv.encrypt(im1, pk)
        pim1 = bgv.decrypt(cim1,sk)
        # print('first plaintext %s'%pim1)
        cim2 = bgv.encrypt(im2, pk)
        pim2 = bgv.decrypt(cim2, sk)
        # print('second plaintext %s'%pim2)
        # print('ring mult %s'%(pim1*pim2))
        # print(cirmod)
        cir = bgv.mul(cim1,cim2)
        plaincir = bgv.decrypt(cir, sk)

        n = 5
        t = 11
        g = 4
        in1 = [1,1,0,1,1]
        in2 = [1,0,0,0,1]
        inv1 = inverse_transform(in1,g, t)
        inv2 = inverse_transform(in2,g, t)

        cir = circular_convolve(inv1,inv2)
        cirmod = [i%t for i in cir]

        compwise = transform(cirmod, g, t)

        self.assertEqual(compwise,[1,0,0,0,1])
        # print(im1*im2)
        # encrypt the new encoding
        # cim1 = bgv.encrypt(im1,pk)
        # cim2 = bgv.encrypt(im2,pk)
        # # mult homomorphically
        # mulcomp = bgv.mul(cim1, cim2)
        # # decrypt
        # plainmult = bgv.decrypt(mulcomp,sk)
        # plainmult = Rq.positive_q(plainmult, t)
        # # NTT again
        # mult = transform(plainmult, g, t)
        # print(mult)
        
    def test_find_square_root(self):
        n = 4
        k = 10
        q = n*k + 1
        g = find_generator(q-1, q)
        omega = modmath(g**k, q)
        psi = square_root_mod(q, n)
        self.assertEqual(modmath(psi*psi, q), omega)

    def test_set_up_params(self):
        n = 16
        qbits = 8
        q = set_up_params(n, qbits)
        self.assertTrue(is_prime(q))
        self.assertTrue(is_power2(n))
        self.assertTrue((q-1)%(2*n) == 0)


class TestRq(TestCase):
    def test_one(self):
        self.assertEqual(2, 2)

    def test_init(self):
        a = Rq(n=3, q=5, coeffs=[1, 2, 3])
        self.assertEqual(a, [1, 2, -2])

    def test_inner_product_Rq_with_pow_base_and_decomp(self):
        a = Rq(n=3, q=19, coeffs=[12, 13, 14])
        b = Rq(n=3, q=19, coeffs=[1, 2, 3])
        c = Rq.inner_product(a, b, 19)
        self.assertEqual(c, 4)
        decomp_a = decomp(a, a.q)
        decomp_a = extract_list_ring(decomp_a)
        b_pow = pow_base(b, b.q)
        b_pow = extract_list_ring(b_pow)
        d = Rq.inner_product(decomp_a, b_pow, 19)
        self.assertEqual(d, 4)

    def test_pow_base(self):
        a = Rq(n=3, q=19, coeffs=[1, 2, 3])
        a_pow = pow_base(a, a.q)
        # print(a_pow)

    def test_bit_decomp(self):
        a = Rq(n=3, q=5, coeffs=[3, 4, 18])
        decomp_a = decomp(a, a.q)
        self.assertEqual(decomp_a, [[1,0,1],[1,0,1],[0,1,0]])

    def test_base_operation(self):
        a = 1002
        b = base(a,2)
        self.assertEqual(b, [0, 1, 0, 1, 0, 1 , 1, 1, 1, 1])
        b = base(a,5)
        self.assertEqual(b, [2, 0, 0, 3, 1])
        c = base(a,32)
        self.assertEqual(c, [10, 31])
        b = base(a,16)
        self.assertEqual(b, [10, 14, 3])

    def test_flatten_list_of_ring_element(self):
        a = Rq(n=3, q=13, coeffs=[100, 200, 300])
        b = Rq(n=3, q=13, coeffs=[10, 20, 30])
        c = [a, b]
        g = Gadget(3, 4)
        exp_c = g.forward(c)
        d = g.backward(exp_c)
        e = Rq(n = 3, q = 13, coeffs = d[0])
        f = Rq(n = 3, q = 13, coeffs = d[1])
        h = [e,f]
        self.assertEqual(c, h)

    def test_flatten_ring_element(self):
        a = Rq(n=3, q=13, coeffs=[100, 200, 300])
        b = [9, 5, 1]
        g = Gadget(3, 4)
        exp_a = g.forward(a)
        exp_b = g.forward(b)
        self.assertEqual(exp_a, exp_b)
        c = g.backward(exp_a)
        c = Rq(n = 3, q = 13, coeffs = c)
        self.assertEqual(c, a)

    def test_flatten_array_of_array_of_int(self):
        a = [[1002,1002,1002],[1002,1002,1002],[1002,1002,1002],[1002,1002,1002]]
        g = Gadget(16, 3)
        exp_a = g.forward(a)
        self.assertEqual(exp_a, [[[10,14,3],[10,14,3],[10,14,3]],[[10,14,3],[10,14,3],[10,14,3]],[[10,14,3],[10,14,3],[10,14,3]],[[10,14,3],[10,14,3],[10,14,3]]])
        b = g.backward(exp_a)
        self.assertEqual(b, a)

    def test_flatten_array_of_int(self):
        a = [1002, 1002, 1002]
        g = Gadget(16, 3)
        exp_a = g.forward(a)
        self.assertEqual(exp_a, [[10,14,3],[10,14,3],[10,14,3]])
        b = g.backward(exp_a)
        self.assertEqual(b, a)


    def test_flatten_gadget(self):
        # a is an integer
        a = 1002
        b = 2
        g = Gadget(base=b, length = 15)
        exp_a = g.forward(a)
        self.assertEqual(exp_a, [0,1,0,1,0,1,1,1,1,1,0,0,0,0,0])
        b = g.backward(exp_a)
        self.assertEqual(b, 1002)
        b = 5
        g = Gadget(base=b, length = 5)
        exp_a = g.forward(a)
        self.assertEqual(exp_a, [2, 0, 0, 3, 1])
        b = g.backward(exp_a)
        self.assertEqual(b, 1002)
        b = 16
        g = Gadget(base=b, length = 5)
        exp_a = g.forward(a)
        self.assertEqual(exp_a, [10,14,3,0,0])
        b = g.backward(exp_a)
        self.assertEqual(b, 1002)

    def test_mod_math(self):
        a = 0
        b = 5
        self.assertEqual(0, modmath(a, b))
        self.assertEqual(1, modmath(a + 1, b))
        self.assertEqual(2, modmath(a + 2, b))
        self.assertEqual(-2, modmath(a + 3, b))
        self.assertEqual(-1, modmath(a + 4, b))
        self.assertEqual(0, modmath(a + 5, b))

    def test_init_Rq(self):
        a = Rq(n=3, q=5, coeffs=[1, 2, 3])
        self.assertEqual(a, [1, 2, -2])
        a = Rq(n=3, q=5, coeffs=[1, 2, 5])
        self.assertEqual(a, [1, 2, 0])
        a = Rq(n=3, q=5, coeffs=[1, 2, 6])
        self.assertEqual(a, [1, 2, 1])
        a = Rq(n=3, q=5, coeffs=[1, 2, 7])
        self.assertEqual(a, [1, 2, 2])
        a = Rq(n=3, q=5, coeffs=[1, 2, 8])
        self.assertEqual(a, [1, 2, -2])
        a = Rq(n=3, q=5, coeffs=[1, 2, 9])
        self.assertEqual(a, [1, 2, -1])
        a = Rq(n=3, q=5, coeffs=[1, 2, 10])
        self.assertEqual(a, [1, 2, 0])
        a = Rq(n=3, q=5, coeffs=[1, 2, 11])
        self.assertEqual(a, [1, 2, 1])
        a = Rq(n=3, q=5, coeffs=[-1, 2, 11])
        self.assertEqual(a, [-1, 2, 1])
        a = Rq(n=3, q=5, coeffs=[-2, 2, 11])
        self.assertEqual(a, [-2, 2, 1])
        a = Rq(n=3, q=5, coeffs=[-3, 2, 11])
        self.assertEqual(a, [2, 2, 1])
        a = Rq(n=3, q=5, coeffs=[-4, 2, 11])
        self.assertEqual(a, [1, 2, 1])
        a = Rq(n=3, q=5, coeffs=[-5, 2, 11])
        self.assertEqual(a, [0, 2, 1])
        a = Rq(n=3, q=5, coeffs=[1, 1, 1, 1])
        self.assertEqual(a, [0, 1, 1])
        a = Rq(n=3, q=5, coeffs=[1, 1, 1, 1, -7])
        self.assertEqual(a, [0, -2, 1])

    def test_assign_Rq(self):
        a = Rq(n=3, q=5, coeffs=[1, 2, 3])
        a[0] = 2
        self.assertEqual(a, [2, 2, -2])
        a[0] = 5
        self.assertEqual(a, [0, 2, -2])
        a[0] = 6
        self.assertEqual(a, [1, 2, -2])
        a[0] = 9
        self.assertEqual(a, [-1, 2, -2])
        a[0] = -1
        self.assertEqual(a, [-1, 2, -2])

    def test_add(self):
        a = Rq(n=3, q=5, coeffs=[1, 2, 3])
        b = Rq(n=3, q=5, coeffs=[1, 1, 1])
        c = a + b
        self.assertEqual(c, [2, -2, -1])
        self.assertEqual(a, [1, 2, -2])
        self.assertEqual(b, [1, 1, 1])
        a = Rq(n=3, q=5, coeffs=[1, 2, 2])
        b = Rq(n=3, q=5, coeffs=[2, 2, 2])
        c = a + b
        self.assertEqual(c, [-2, -1, -1])

    def test_substract(self):
        a = Rq(n=3, q=5, coeffs=[1, 2, 3])
        b = Rq(n=3, q=5, coeffs=[1, 1, 1])
        c = a - b
        self.assertEqual(c, [0, 1, 2])
        self.assertEqual(a, [1, 2, -2])
        self.assertEqual(b, [1, 1, 1])
        a = Rq(n=3, q=5, coeffs=[1, 2, 2])
        b = Rq(n=3, q=5, coeffs=[2, 2, 2])
        c = a - b
        self.assertEqual(c, [-1, 0, 0])
        a = Rq(n=3, q=5, coeffs=[0, 0, 0])
        b = Rq(n=3, q=5, coeffs=[2, 2, 2])
        c = a - b
        self.assertEqual(c, [-2, -2, -2])

    def test_poly_mult(self):
        a = [1, 2, 3]
        b = [1, 1, 1]
        c = poly_multiply(a, b)
        self.assertEqual(c, [1, 3, 6, 5, 3])

    def test_mult(self):
        a = Rq(n=3, q=5, coeffs=[1, 2, 3])
        b = Rq(n=3, q=5, coeffs=[1, 1, 1])
        c = a * b
        self.assertEqual(c, [1, 0, 1])
        self.assertEqual(a, [1, 2, -2])
        self.assertEqual(b, [1, 1, 1])
        a = Rq(n=3, q=5, coeffs=[1, 2, 2])
        b = Rq(n=3, q=5, coeffs=[2, 2, 2])
        c = a * b
        self.assertEqual(c, [-1, 2, 0])
        a = Rq(n=3, q=5, coeffs=[1, 1, 1])
        b = Rq(n=3, q=5, coeffs=[2, 2, 2])
        c = Rq(n=3, q=5, coeffs=[0, 0, 0])
        # test scalar product and some more complicated statement
        d = c - (a * a + b * b)
        self.assertEqual(d, [0, 0, 0])
        self.assertEqual(2 * a, [2, 2, 2])
        self.assertEqual(3 * a, [-2, -2, -2])
        self.assertEqual(4 * a, [-1, -1, -1])
        self.assertEqual(5 * a, [0, 0, 0])
        # test exponential
        self.assertEqual(b ** 2, [1, -1, 2])
        self.assertEqual(b ** 0, [1, 1, 1])
        self.assertEqual(b ** 3, [0, 1, -1])
        b = Rq(n=3, q=5, coeffs=[1, 2, 3])
        self.assertEqual(b ** 2, [-1, 0, 0])
        self.assertEqual(b ** 3, [-1, -2, 2])
        self.assertEqual(b ** 0, [1, 1, 1])
        # test multiply with 1 scalar
        a = 1
        b = Rq(n=3, q=5, coeffs=[2, 2, 2])
        self.assertEqual(a * b, [2, 2, 2])
        self.assertEqual(b * a, [2, 2, 2])

    def test_ring_definition(self):
        # test associative, commutative, distributive
        n = 3
        q = 5
        a = Rq.random_samples(n, q)
        b = Rq.random_samples(n, q)
        c = Rq.random_samples(n, q)
        zero = Rq(n=3, q=5, coeffs=[0, 0, 0])
        one = Rq(n=3, q=5, coeffs=[1, 0, 0])

        self.assertEqual((a + b) + c, a + (b + c))
        self.assertEqual(a + b, b + a)
        self.assertEqual(a - a, zero)
        self.assertEqual(a + zero, a)
        self.assertEqual(a * b, b * a)
        self.assertEqual((a * b) * c, a * (b * c))
        self.assertEqual((a * one), (one * a))
        self.assertEqual(a * (b + c), (a * b) + (a * c))
        self.assertEqual((b + c) * a, (b * a) + (c * a))
        self.assertEqual(a * zero, zero)
        self.assertEqual(zero * a, zero)

    def test_hamming_distance_plain(self):
        n = 3
        q = 5
        a = Rq(n=3, q=5, coeffs=[0, 0, 0])
        b = Rq(n=3, q=5, coeffs=[1, 0, 0])
        self.assertEqual(Rq.hd_plain(a,b),1)
        a = Rq(n=3, q=5, coeffs=[0, 1, 0])
        self.assertEqual(Rq.hd_plain(a,b),2)
        a = Rq(n=3, q=5, coeffs=[0, 1, 1])
        self.assertEqual(Rq.hd_plain(a,b),3)


class TestBV(TestCase):
    def setUp(self):
        self.n = 3
        self.q = 40433
        self.sigma = 4
        self.t = 2
        self.bv = BV(n=self.n, q=self.q, t=self.t, sigma=self.sigma)

    def test_genkey(self):
        n = self.bv.n
        q = self.bv.q
        sigma = self.bv.sigma

        self.assertEqual(self.bv.q, 40433)
        small = small_samples(n, sigma)
        for i in small:
            self.assertLessEqual(i, sigma)
        large = large_samples(n, q)

        (sk, pk) = self.bv.genkey()

        te = (pk[1] * sk + pk[0])
        for i in te:
            self.assertLess(i, 3 * sigma)

    def test_enc(self):
        (sk, pk) = self.bv.genkey()
        m = Rq(n=3, q=self.t, coeffs=[1, 1, 0])
        c = self.bv.enc(m, pk)
        plain = self.bv.dec(c, sk)
        self.assertEqual(m, plain)

    def test_homomorphic_add(self):
        (sk, pk) = self.bv.genkey()
        m1 = Rq(n=3, q=self.q, coeffs=[1, 1, 0])
        m2 = Rq(n=3, q=self.q, coeffs=[1, 1, 1])
        mSum = Rq(n=3, q=self.q, coeffs=[0, 0, 1])
        c1 = self.bv.enc(m1, pk)
        c2 = self.bv.enc(m2, pk)
        cSum = self.bv.add(c1, c2)
        plainSum = self.bv.dec(cSum, sk)
        self.assertEqual(plainSum, mSum)

    def test_pad_ciphertext(self):
        m1 = Rq(n=3, q=self.q, coeffs=[1, 1, 1])
        c1 = [m1, m1]
        c2 = [m1, m1, m1, m1]
        self.bv.pad_ring(c1, c2)
        self.assertEqual(len(c1), len(c2))
        c2 = [m1, m1]
        c1 = [m1, m1, m1, m1]
        self.bv.pad_ring(c1, c2)
        self.assertEqual(len(c1), len(c2))
        c2 = [m1, m1]
        c1 = [m1, m1]
        self.bv.pad_ring(c1, c2)
        self.assertEqual(len(c1), len(c2))

    def test_cross_product(self):
        m1 = Rq(n=3, q=self.q, coeffs=[1, 1, 1])
        m2 = Rq(n=3, q=self.q, coeffs=[1, 1, 2])
        m3 = Rq(n=3, q=self.q, coeffs=[1, 1, 3])
        m4 = Rq(n=3, q=self.q, coeffs=[1, 1, 4])
        c1 = [m1, m2]
        c2 = [m3, m4]
        c3 = self.bv.mult(c1, c2)
        self.assertEqual(c3[0], m1 * m3)
        self.assertEqual(c3[1], m2 * m3 + m1 * m4)
        self.assertEqual(c3[2], m2 * m4)
        m1 = Rq(n=3, q=self.q, coeffs=[1, 1, 1])
        m2 = Rq(n=3, q=self.q, coeffs=[1, 1, 2])
        m3 = Rq(n=3, q=self.q, coeffs=[1, 1, 3])
        m4 = Rq(n=3, q=self.q, coeffs=[1, 1, 4])
        m5 = Rq(n=3, q=self.q, coeffs=[1, 1, 5])
        m6 = Rq(n=3, q=self.q, coeffs=[1, 1, 6])
        c1 = [m1, m2, m3]
        c2 = [m4, m5, m6]
        c3 = self.bv.mult(c1, c2)
        self.assertEqual(c3[0], m1 * m4)
        self.assertEqual(c3[1], m2 * m4 + m1 * m5)
        self.assertEqual(c3[2], m3 * m4 + m2 * m5 + m1 * m6)
        self.assertEqual(c3[3], m3 * m5 + m2 * m6)
        self.assertEqual(c3[4], m3 * m6)
        m1 = Rq(n=3, q=self.q, coeffs=[1, 1, 1])
        m2 = Rq(n=3, q=self.q, coeffs=[1, 1, 2])
        m3 = Rq(n=3, q=self.q, coeffs=[1, 1, 3])
        m4 = Rq(n=3, q=self.q, coeffs=[1, 1, 4])
        m5 = Rq(n=3, q=self.q, coeffs=[1, 1, 5])
        m6 = Rq(n=3, q=self.q, coeffs=[1, 1, 6])
        m7 = Rq(n=3, q=self.q, coeffs=[1, 1, 7])
        m8 = Rq(n=3, q=self.q, coeffs=[1, 1, 8])
        c1 = [m1, m2, m3, m4]
        c2 = [m5, m6, m7, m8]
        c3 = self.bv.mult(c1, c2)
        self.assertEqual(c3[0], m1 * m5)
        self.assertEqual(c3[1], m2 * m5 + m1 * m6)
        self.assertEqual(c3[2], m3 * m5 + m1 * m7 + m2 * m6)
        self.assertEqual(c3[3], m4 * m5 + m3 * m6 + m2 * m7 + m1 * m8)
        self.assertEqual(c3[4], m4 * m6 + m3 * m7 + m2 * m8)
        self.assertEqual(c3[5], m3 * m8 + m4 * m7)
        self.assertEqual(c3[6], m4 * m8)
        self.assertEqual(len(c3), 7)

    def test_homomorphic_mult(self):
        (sk, pk) = self.bv.genkey()
        m1 = Rq(n=3, q=self.q, coeffs=[1, 1, 0])
        m2 = Rq(n=3, q=self.q, coeffs=[1, 1, 1])
        mmul = Rq(n=3, q=self.q, coeffs=[0, 0, 0])
        c1 = self.bv.enc(m1, pk)
        c2 = self.bv.enc(m2, pk)
        p1 = self.bv.dec(c1, sk)
        p2 = self.bv.dec(c2, sk)
        self.assertEqual(m1, p1)
        self.assertEqual(m2, p2)
        cmul = self.bv.mult(c1, c2)
        plainmul = self.bv.dec(cmul, sk)
        self.assertEqual(plainmul, mmul)

    def test_add_mult_diff_t(self):
        # t and sigma choices affect the error growth, which then
        # affect the choice of q
        bv2 = BV(n=3, t=20, q=2 ** 100 - 3)
        (sk, pk) = bv2.genkey()
        m1 = Rq(n=3, q=bv2.t, coeffs=[15, 4, 1])
        m2 = Rq(n=3, q=bv2.t, coeffs=[1, 2, 8])
        c1 = bv2.enc(m1, pk)
        c2 = bv2.enc(m2, pk)
        p1 = bv2.dec(c1, sk)
        p2 = bv2.dec(c2, sk)
        self.assertEqual(m1, p1)
        self.assertEqual(m2, p2)
        c_add = bv2.add(c1, c2)
        p_add = bv2.dec(c_add, sk)
        self.assertEqual(p_add, m1 + m2)
        c_mul = bv2.mult(c1, c2)
        p_mul = bv2.dec(c_mul, sk)
        self.assertEqual(p_mul, m1 * m2)

    def test_add_mult_diff_n(self):
        # t and sigma choices affect the error growth, which then
        # affect the choice of q
        bv2 = BV(n=4, t=10, q=2 ** 20 - 3)
        (sk, pk) = bv2.genkey()
        m1 = Rq(n=4, q=bv2.t, coeffs=[15, 4, 1, 2])
        m2 = Rq(n=4, q=bv2.t, coeffs=[1, 2, 8, 5])
        c1 = bv2.enc(m1, pk)
        c2 = bv2.enc(m2, pk)
        p1 = bv2.dec(c1, sk)
        p2 = bv2.dec(c2, sk)
        self.assertEqual(m1, p1)
        self.assertEqual(m2, p2)
        c_add = bv2.add(c1, c2)
        p_add = bv2.dec(c_add, sk)
        self.assertEqual(p_add, m1 + m2)
        c_mul = bv2.mult(c1, c2)
        p_mul = bv2.dec(c_mul, sk)
        self.assertEqual(p_mul, m1 * m2)

    def test_pack1_pack2(self):
        bv = BV(n=4, t=20, q=2 ** 20 - 3)
        (sk, pk) = bv.genkey()
        m = Rq(n=4, q=bv.t, coeffs=[1, 1, 1, 0])
        pm1 = bv.pack1(m)
        self.assertEqual(m, pm1)
        m2 = Rq(n=4, q=bv.t, coeffs=[1, 0, -1, -1])
        pm2 = bv.pack2(m)
        self.assertEqual(m2, pm2)

    def test_inner_product(self):
        bv = BV(n=6, t=10, q=2 ** 100 - 3)
        (sk, pk) = bv.genkey()
        p = Rq(n=6, q=bv.t, coeffs=[1, 1, 0, 0, 1, 1])
        q = Rq(n=6, q=bv.t, coeffs=[1, 1, 0, 1, 0, 1])

        pm1 = bv.pack1(p)
        pm2 = bv.pack2(q)
        mul = pm1 * pm2

        c1 = bv.enc(pm1, pk)
        c2 = bv.enc(pm2, pk)
        c = bv.dec(bv.mult(c1, c2), sk)
        self.assertEqual(c, mul)
        self.assertEqual(c[0], 3)

    def test_hamming_distance(self):
        bv = BV(n=6, t=12, q=2 ** 100 - 3)
        (sk, pk) = bv.genkey()
        p = Rq(n=6, q=bv.t, coeffs=[1, 0, 1, 0, 1, 0])
        q = Rq(n=6, q=bv.t, coeffs=[1, 1, 0, 1, 0, 1])
        p = bv.pack1(p)
        q = bv.pack2(q)
        encp = bv.enc(p, pk)
        encq = bv.enc(q, pk)
        hd = bv.compute_hd(encp, encq)
        plain_hd = bv.dec(hd, sk)
        self.assertEqual(plain_hd[0], 5)

    def test_mask_hd(self):
        bv = BV(n=6, t=12, q=2 ** 100 - 3)
        (sk, pk) = bv.genkey()
        p = Rq(n=6, q=bv.t, coeffs=[1, 0, 1, 0, 1, 0])
        q = Rq(n=6, q=bv.t, coeffs=[1, 1, 0, 1, 0, 1])
        p = bv.pack1(p)
        q = bv.pack2(q)
        encp = bv.enc(p, pk)
        encq = bv.enc(q, pk)
        hd = bv.compute_hd(encp, encq)
        plain_hd = bv.dec(hd, sk)
        self.assertEqual(plain_hd[0], 5)
        one = bv.enc(bv.one, pk)
        mask = bv.add(one, hd)
        dec_mask = bv.dec(mask, sk)
        self.assertEqual(dec_mask[0], 6)

    def test_unary_encode(self):
        bv = BV(n=6, t=12, q=2 ** 25 - 3)

        # input = 1
        input = bv.n // 2 + 1
        rinput = bv.unary_encode(input)
        a = [0 for _ in range(0, bv.n)]
        a[input] = 1

        self.assertEqual(a, rinput)

    def test_unary_encode_m2(self):
        bv = BV(n=6, t=12, q=2 ** 25 - 3)

        input = bv.n
        rinput = bv.unary_encode(input)

        a = [0 for _ in range(0, bv.n)]
        a[0] = -1

        self.assertEqual(a, rinput)

    def test_encrypt_type_2(self):
        bv = BV(n=6, t=12, q=2 ** 25 - 3)
        x = 1
        m = bv.unary_encode(x)
        (sk, pk) = bv.genkey()
        c = bv.enc(m, pk)
        p = bv.dec(c, sk)
        y = bv.unary_decode(p)
        self.assertEqual(x, y)

        x = 11
        m = bv.unary_encode(x)
        (sk, pk) = bv.genkey()
        c = bv.enc(m, pk)
        p = bv.dec(c, sk)
        y = bv.unary_decode(p)
        self.assertEqual(x, y)

    def test_inner_product_Rq(self):
        a = Rq(n=3, q=5, coeffs=[1, 1, 1])
        b = Rq(n=3, q=5, coeffs=[1, 1, 1])
        c = Rq.inner_product(a, b, 5)
        self.assertEqual(c, -2)

    def test_rot_matrix(self):
        a = Rq(n=3, q=5, coeffs=[1, 1, 1])
        b = a.rot_matrix()
        self.assertEqual(b, [[1, 1, 1], [-1, 1, 1], [-1, -1, 1]])
        a = Rq(n=3, q=5, coeffs=[1, 2, 3])
        b = a.rot_matrix()
        self.assertEqual(b, [[1, 2, -2], [2, 1, 2], [-2, 2, 1]])
        c = Rq.vec_mult_matrix(a, b, 5)
        # print(c)

    def test_mult_matrix(self):
        a = Rq(n=3, q=5, coeffs=[1, 1, 1])
        b = a.rot_matrix()
        c = Rq.vec_mult_matrix(a, b, a.q)
        # print(c)

    def test_msb_extract(self):
        bv = BV(n=3, t=6, q=113, sigma=1)
        x = 5
        m = bv.unary_encode(x)
        # print(m)
        (sk, pk) = bv.genkey()
        c = bv.enc(m, pk)
        # print(plain_m)
        lwe_c = bv.to_lwe(c)
        # print(lwe_c)
        plain = bv.decrypt_lwe(lwe_c, sk)
        self.assertEqual(plain,-1)

    def test_msb_extract2(self):
        bv = BV(n=100, t=100, q=2**100, sigma=1)
        x = 90
        m = bv.unary_encode(x)
        # print(m)
        (sk, pk) = bv.genkey()
        c = bv.enc(m, pk)
        lwe_c = bv.to_lwe(c)
        # print(lwe_c)
        plain = bv.decrypt_lwe(lwe_c, sk)
        self.assertEqual(plain,1)

    def test_add_matrix(self):
        m1 = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        m2 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        t = Rq.add_matrix(m1,m2,5)
        self.assertEqual(t, [[2,2,2],[-2,-2,-2],[-1,-1,-1]])

    def test_rot_add(self):
        # rot(a) = rot(b) + rot(c) if a = b + c
        b = Rq(3, 5, [1,1,1])
        c = Rq(3, 5, [2,2,2])
        a = b + c
        self.assertEqual(a.rot_matrix(),
                         (Rq.add_matrix(b.rot_matrix(), c.rot_matrix(),5)))
        a = b*c
        # print(a.rot_matrix())
        # with Timer() as t:
        #     a = b*c
        # print('time is %s ms'%t.msecs)

    def test_decryption_of_ufg_zero(self):
        bv = BV(n = 3, q = 2**10, t = 6, sigma= 1)
        X = [0 for _ in range(bv.n)]
        X[0] = -1
        X[1] = 1
        X = Rq(bv.n, bv.t, X)
        (sk,pk) = bv.genkey()
        c = [X, bv.zeros]
        p = bv.dec(c, sk)
        self.assertEqual(p,[-1,1,0])
        X = [0 for _ in range(bv.n)]
        X[0] = 1
        X = Rq(bv.n, bv.t, X)
        c = [X, bv.zeros]
        p = bv.dec(c, sk)
        self.assertEqual(p, [1, 0, 0])

    def test_first_term(self):
        bv = BV(n=5, q=2 ** 80, t=10, sigma=1)
        X = [0 for _ in range(bv.n)]
        X[0] = -1
        X[1] = 1
        X = Rq(bv.n, bv.t, X)
        (sk,pk) = bv.genkey()
        encx = bv.enc(X, pk)
        Y = [0 for _ in range(bv.n)]
        Y[0] = 1
        Y = Rq(n = bv.n, q = bv.q, coeffs=Y)
        encb = bv.enc(Y, pk)
        mul = bv.mult(encx,encb)
        p = bv.dec(mul, sk)
        self.assertEqual(p,X)
        enc1 = bv.enc(bv.one,pk)
        r = bv.add(mul,enc1)
        p = bv.dec(r, sk)
        encb = bv.enc(bv.zeros, pk)
        mul = bv.mult(encx, encb)
        p = bv.dec(mul, sk)
        enc1 = bv.enc(bv.one, pk)
        r = bv.add(mul, enc1)
        p = bv.dec(r, sk)
        self.assertEqual(bv.one, p)

    def test_negative_hd(self):
        bv = BV(n=30, q=2 ** 80, t=30, sigma=1)
        (sk, pk) = bv.genkey()
        hd = small_samples(bv.t, bv.sigma)
        hd[0] = 17
        hd = Rq(bv.n, bv.t, hd)

        enc_hd = bv.enc(hd, pk)
        pos = bv.dec(enc_hd,sk)
        enc_minus_hd = bv.sub([bv.zeros,bv.zeros],enc_hd)
        neg = bv.dec(enc_minus_hd,sk)
        self.assertEqual(pos[0],-neg[0])

    def test_HD_by_mult(self):
        bv = BV(n=30, q=2 ** 80, t=30, sigma=1)
        (sk,pk) = bv.genkey()
        hd = 17
        l = floor(log(hd, 2)) - 1
        #decompose into bints
        hd_bits = bv.bit_repr(hd,l).to01()
        # print(hd_bits)
        list_bits = []
        for i, j in enumerate(hd_bits):
            temp = [0 for _ in range(bv.n)]
            temp[0] = int(hd_bits[i])
            list_bits.append(Rq(n=bv.n, q=bv.q, coeffs=temp))

        enc_b = []
        for i in list_bits:
            C_i = bv.enc(i, pk)
            enc_b.append(C_i)

        #converting
        enc_x_pow_jb = []
        for i,j in enumerate(enc_b):
            c = bv.map_to_j(enc_b[i],2**i)
            enc_x_pow_jb.append(c)

        # #multiply them together
        enc_hd = bv.bin_tree_mult(enc_x_pow_jb)
        plain = bv.unary_decode((bv.dec(enc_hd,sk)))
        self.assertEqual(plain, hd)

    def test_negative_HD_by_mult(self):
        bv = BV(n=50, q=2 ** 80, t=50, sigma=1)
        (sk, pk) = bv.genkey()
        hd = 17
        l = floor(log(hd, 2)) - 1
        # decompose into bins
        hd_bits = bv.bit_repr(hd, l).to01()
        # print(hd_bits)
        list_bits = []
        for i, j in enumerate(hd_bits):
            temp = [0 for _ in range(bv.n)]
            temp[0] = -int(hd_bits[i])
            list_bits.append(Rq(n=bv.n, q=bv.q, coeffs=temp))

        enc_b = []
        for i in list_bits:
            C_i = bv.enc(i, pk)
            enc_b.append(C_i)

        # converting
        enc_x_pow_jb = []
        for i, j in enumerate(enc_b):
            c = bv.map_to_j_neg(enc_b[i], 2 ** i)
            enc_x_pow_jb.append(c)

        # #multiply them together
        enc_hd = bv.bin_tree_mult(enc_x_pow_jb)
        plain = bv.dec(enc_hd,sk)
        # print(plain)
        # print(bv.unary_decode(plain))
        # plain = bv.unary_decode((bv.dec(enc_hd, sk)))
        # self.assertEqual(plain, hd)

        Co = bv.unary_encode(41)
        Enc_co = bv.enc(Co, pk)
        mult = bv.mult(Enc_co, enc_hd)
        plain = bv.dec(mult, sk)


    def test_bin_to_unary(self):
        bv = BV(n = 10, q = 2**60, t = 20, sigma= 1)
        (sk,pk) = bv.genkey()
        cbin = bv.enc(bv.one,pk)
        cbin1 = bv.map_to_j(cbin,1)
        decbin1 = bv.dec(cbin1,sk)
        self.assertEqual(decbin1[1],1)
        cbin = bv.enc(bv.zeros, pk)
        cbin1 = bv.map_to_j(cbin,1)
        decbin1 = bv.dec(cbin1, sk)
        self.assertEqual(decbin1[0], 1)
        cbin = bv.enc(bv.one, pk)
        cbin2 = bv.map_to_j(cbin, 2)
        decbin2 = bv.dec(cbin2, sk)
        self.assertEqual(decbin2[2], 1)
        cbin = bv.enc(bv.zeros, pk)
        cbin2 = bv.map_to_j(cbin, 2)
        decbin2 = bv.dec(cbin2, sk)
        self.assertEqual(decbin2[0], 1)



    def test_map_to_minus_j_power(self):
        bv = BV(n=10, q=2 ** 60, t=20, sigma=1)
        (sk, pk) = bv.genkey()
        m_one = list(bv.one)
        m_one[0] = 0
        cbin = bv.enc(m_one,pk)
        cbin1 = bv.map_to_j_neg(cbin, 1)
        plain = bv.dec(cbin1,sk)
        self.assertEqual(plain[0], 1)
        m_one[0] = -1
        cbin = bv.enc(m_one, pk)
        cbin1 = bv.map_to_j_neg(cbin, 1)
        plain = bv.dec(cbin1, sk)
        self.assertEqual(plain[bv.n-1], -1)
        m_one[0] = 0
        cbin = bv.enc(m_one, pk)
        cbin1 = bv.map_to_j_neg(cbin, 4)
        plain = bv.dec(cbin1, sk)
        self.assertEqual(plain[0], 1)
        m_one[0] = -1
        cbin = bv.enc(m_one, pk)
        cbin1 = bv.map_to_j_neg(cbin, 4)
        plain = bv.dec(cbin1, sk)
        self.assertEqual(plain[bv.n - 4], -1)


    def test_basic_mult_property(self):
        bv = BV(n = 10, q = 2**80, t = 20, sigma= 1)
        (sk,pk) = bv.genkey()
        m = bv.small_samples()
        c = bv.enc(m, sk)
        one = bv.enc(bv.one, pk)
        zero = bv.enc(bv.zeros, pk)
        # multiplying with one
        p1 = bv.mult(c, one)
        p2 = bv.mult(one, c)
        r1 = bv.dec(p1, sk)
        r2 = bv.dec(p2, sk)
        self.assertEqual(r1, r2)

        # mutliplying with zero
        p1 = bv.mult(c, zero)
        r1 = bv.dec(p1, sk)
        self.assertEqual(r1, bv.zeros)
        p1 = bv.mult(zero, c)
        r1 = bv.dec(p1, sk)
        self.assertEqual(r1, bv.zeros)

        m1 = bv.small_samples()
        m2 = bv.small_samples()
        m3 = bv.small_samples()
        c1 = bv.enc(m1, pk)
        c2 = bv.enc(m2, pk)
        c3 = bv.enc(m3, pk)

        # commutative
        c_sum_1 = bv.add(c1, c2)
        c_sum_2 = bv.add(c2, c1)
        self.assertEqual(bv.dec(c_sum_1,sk),bv.dec(c_sum_2,sk))
        c_mult_1 = bv.mult(c1, c2)
        c_mult_2 = bv.mult(c2, c1)
        self.assertEqual(bv.dec(c_mult_1,sk),bv.dec(c_mult_2,sk))

        #associative
        c_sum_3 = bv.add(c_sum_1,c3)
        c_sum_4 = bv.add(c2, c3)
        c_sum_5 = bv.add(c1, c_sum_4)
        self.assertEqual(bv.dec(c_sum_3,sk),bv.dec(c_sum_5,sk))
        c_pro_3 = bv.mult(c_mult_1, c3)
        c_pro_4 = bv.mult(c2, c3)
        c_pro_5 = bv.mult(c1, c_pro_4)
        self.assertEqual(bv.dec(c_pro_3,sk),bv.dec(c_pro_5,sk))

        #distributive
        #c1(c2+c3) = c1.c2 + c1.c3
        term1 = bv.mult(c1, bv.add(c2,c3))
        term2 = bv.add(bv.mult(c1,c2),bv.mult(c1,c3))
        self.assertEqual(bv.dec(term1,sk),bv.dec(term2,sk))

    def test_noise_growth(self):
        bv = BV(n=100, q=2 ** 80, t=1000, sigma=4)
        (sk, pk) = bv.genkey()
        m1 = Rq(bv.n, bv.t, small_samples(bv.t, bv.sigma))
        m2 = Rq(bv.n, bv.t, small_samples(bv.t, bv.sigma))
        m3 = Rq(bv.n, bv.t, small_samples(bv.t, bv.sigma))
        m4 = Rq(bv.n, bv.t, small_samples(bv.t, bv.sigma))
        m5 = Rq(bv.n, bv.t, small_samples(bv.t, bv.sigma))
        m6 = Rq(bv.n, bv.t, small_samples(bv.t, bv.sigma))

        c1 = bv.enc(m1, pk)
        p1 = bv.dec(c1, sk)
        # print('noise for 1 fresh ciphertext %s'%bv.last_noise)
        c2 = bv.enc(m2, pk)
        mult1 = bv.mult(c1,c2)
        dec_mult1 = bv.dec(mult1, sk)
        # print('noise for 1st level of mult %s'%bv.last_noise)

        c3 = bv.enc(m3, pk)
        mult2 = bv.mult(mult1, c3)
        dec_mult2 = bv.dec(mult2, sk)
        # print('noise for 1 extra mult %s'%bv.last_noise)
        # print('q is %s'%bv.q)

        c4 = bv.enc(m4, pk)
        c5 = bv.enc(m5, pk)
        c6 = bv.enc(m6, pk)

    def test_many_multiplication(self):
        bv = BV(n=10, q=2 ** 80, t=10, sigma=2)
        (sk, pk) = bv.genkey()
        m1 = Rq(bv.n, bv.t, small_samples(bv.n,bv.sigma))
        m2 = Rq(bv.n, bv.t, small_samples(bv.n,bv.sigma))
        m3 = Rq(bv.n, bv.t, small_samples(bv.n,bv.sigma))
        m4 = Rq(bv.n, bv.t, small_samples(bv.n,bv.sigma))
        m5 = Rq(bv.n, bv.t, small_samples(bv.n,bv.sigma))
        m6 = Rq(bv.n, bv.t, small_samples(bv.n,bv.sigma))


        c1 = bv.enc(m1, pk)
        c2 = bv.enc(m2, pk)
        c3 = bv.enc(m3, pk)
        c4 = bv.enc(m4, pk)
        c5 = bv.enc(m5, pk)
        c6 = bv.enc(m6, pk)

        #check noise when multiply by bin tree
        with Timer() as ti:
            p1 = bv.mult(c1,c2)
            p2 = bv.mult(c3,c4)
            p3 = bv.mult(p1,p2)
            p4 = bv.mult(p3,c5)
            p5 = bv.mult(p4,c6)
        # print('time to seq mul %s ms'%ti.msecs)
        plain = bv.dec(p5,sk)
        self.assertEqual(m1*m2*m3*m4*m5*m6,plain)

        #check noise when multiply one by one
        with Timer() as ti:
            p1 = bv.mult(c1,c2)
            p2 = bv.mult(p1,c3)
            p3 = bv.mult(p2,c4)
            p4 = bv.mult(p3,c5)
            p5 = bv.mult(p4,c6)
        # print('time to bin mul manually %s ms'%ti.msecs)
        plain = bv.dec(p5, sk)
        self.assertEqual(m1 * m2 * m3 * m4 * m5 * m6, plain)

        c = [c1,c2,c3,c4,c5,c6]
        with Timer() as ti:
            p = bv.bin_tree_mult(c)
        # print('time to bin mul %s ms'%ti.msecs)
        plain = bv.dec(p, sk)
        self.assertEqual(m1 * m2 * m3 * m4 * m5 * m6, plain)

        c = [c1, c2, c3, c4, c5]
        with Timer() as ti:
            p = bv.bin_tree_mult(c)
        # print('time to bin mul %s ms' % ti.msecs)
        plain = bv.dec(p, sk)
        self.assertEqual(m1 * m2 * m3 * m4 * m5, plain)

    def test_rot_mult_1(self):
        bv = BV(n=100, q=2 ** 100, t=100, sigma=2)
        (sk, pk) = bv.genkey()
        one = Rq(n=bv.n, q = bv.q, coeffs=list(bv.one))
        another_one = Rq(n=bv.n, q = bv.q, coeffs=list(bv.one))
        extra_one = Rq(n=bv.n, q = bv.q, coeffs=list(bv.one))


        c_one = bv.enc(one, pk)
        c_another_one = bv.enc(another_one, pk)
        c_extra_one = bv.enc(extra_one, pk)


        cmul = bv.mult(c_one, c_another_one)
        cmul = bv.mult(cmul, c_extra_one)


        # rot_1 = cmul[0].rot_matrix()
        # l0 = Rq.vec_mult_matrix(bv.all_one, rot_1,bv.q)
        # l0 = Rq(n=bv.n, q =bv.q, coeffs=l0)[0]
        #
        # rot_2 = cmul[1].rot_matrix()
        # l1 = Rq.vec_mult_matrix(bv.all_one, rot_2, bv.q)
        # l1 = Rq(n=bv.n, q =bv.q, coeffs=l1)
        #
        # rot_3 = cmul[2].rot_matrix()
        # l2 = Rq.vec_mult_matrix(bv.all_one, rot_3, bv.q)
        # l2 = Rq(n=bv.n, q =bv.q, coeffs=l2)
        #
        # rot_4 = cmul[3].rot_matrix()
        # l3 = Rq.vec_mult_matrix(bv.all_one, rot_4, bv.q)
        # l3 = Rq(n=bv.n, q=bv.q, coeffs=l3)


        # c = [l0, l1, l2, l3]

        c = bv.to_lwe_generic(cmul)

        plain = bv.decrypt_lwe_generic(c, sk)
        self.assertEqual(plain%2,1)
        # print(bv.last_noise)
        # print(rot_1)

    def test_double_crt(self):
        f = [10,20,30]
        from numpy import fft
        # print(fft.fft(f))
        g = [50,60,70]
        n = 3
        q = 385
        f = Rq(n,q,f)
        g = Rq(n, q, g)
        nttf = [f.evaluate_at(1),f.evaluate_at(221),f.evaluate_at(332)]

        h = f*g

if __name__ == '__main__':
    unittest.main()


    # def test_poly_mult_time(self):
    #     bv = BV(n = 100, q = 2**50, t = 500, sigma= 5)
    #     m1 = Rq(n = bv.n, q = bv.t, coeffs=large_samples(bv.n, bv.t))
    #     m2 = Rq(n = bv.n, q = bv.t, coeffs=large_samples(bv.n, bv.t))
    #     with Timer() as t:
    #         poly_multiply(m1,m2)
    #     print('time to do poly multiply %s ms'%t.msecs)
    #     (sk,pk) = bv.genkey()
    #
    #     with Timer() as t:
    #         c = bv.enc(m1, pk)
    #     print('time to do enc %s ms'%t.msecs)
    #
    #     c = bv.enc(m1, pk)
    #     with Timer() as t:
    #         p = bv.dec(c, sk)
    #     print('time to do dec %s ms'%t.msecs)
    #
    #
    #
    #     c1 = bv.enc(m1, pk)
    #     c2 = bv.enc(m2, pk)
    #
    #     with Timer() as t:
    #         p = bv.mult(c1, c2)
    #     print('time to do homo multi %s ms'%t.msecs)
    #
    #     def do_test():
    #         bv.mult(c1, c2)
