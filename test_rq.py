from __future__ import absolute_import

import cProfile
from unittest import TestCase

from bv import poly_multiply, BV, Rq, modmath, \
    small_samples, large_samples, rot
from timer import Timer


class TestRq(TestCase):
    def test_one(self):
        self.assertEqual(2, 2)

    def test_init(self):
        a = Rq(n=3, q=5, coeffs=[1, 2, 3])
        self.assertEqual(a, [1, 2, -2])

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
        plain_m = bv.dec(c, sk)
        # print(plain_m)
        lwe_c = bv.to_lwe(c)
        # print(lwe_c)
        plain = bv.decrypt_lwe(lwe_c, sk)
        self.assertEqual(plain,-1)

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

    def test_bin_to_unary(self):
        bv = BV(n = 10, q = 2**80, t = 20, sigma= 1)
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

    def test_many_multiplication(self):
        bv = BV(n=20, q=2 ** 100, t=40, sigma=3)
        (sk, pk) = bv.genkey()
        m1 = bv.small_samples()
        m2 = bv.small_samples()
        m3 = bv.small_samples()
        m4 = bv.small_samples()

        c1 = bv.enc(m1, pk)
        c2 = bv.enc(m2, pk)
        c3 = bv.enc(m3, pk)
        c4 = bv.enc(m4, pk)
        p1 = bv.mult(c1,c2)
        p2 = bv.mult(c3,c4)
        p3 = bv.mult(p1,p2)
        plain = bv.dec(p3,sk)
        self.assertEqual(m1*m2*m3*m4,plain)
        p1 = bv.mult(c1,c2)
        p2 = bv.mult(p1,c3)
        p3 = bv.mult(p2,c4)
        plain = bv.dec(p3, sk)
        self.assertEqual(m1 * m2 * m3 * m4, plain)

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





