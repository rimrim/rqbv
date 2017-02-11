from random import randint

# note: it's hard to install scipy on windows
from scipy.fftpack import fft, ifft
from numpy import real, base_repr, convolve


def modmath(a, b):
    # type: (int, int) -> int
    c = a % b
    if c > b / 2:
        return c - b
    return c


def rot(a):
    ret = []
    for i in range(len(a)):
        # first column stay the same
        if i == 0:
            ret.append(a)
            continue
        # next columns
        temp = []
        # first element, negate the last element
        # of previous column
        neg = modmath(-ret[i - 1][-1], a.q)
        temp.append(neg)
        # append the others
        for j in range(0, len(a) - 1):
            temp.append(ret[i - 1][j])

        ret.append(temp)
    return ret

def poly_multiply(a, b):
    """multiply 2 polynomials using DFT"""

    return list(convolve(a,b))


def mod_poly(poly, d):
    """divide polynomial by x^d + 1"""
    original_len = len(poly)
    temp = [poly[x] for x in range(0, d)]
    for i in range(d, original_len):
        power = i // d
        index = i % d
        if power == 1:
            plus = -1 * poly[i]
        else:
            plus = ((-1) ** power) * poly[i]
        temp[index] += plus
    # print repr(temp)
    return temp


def small_samples(n, sigma):
    """returns a list"""
    temp = [randint(-sigma, sigma) for x in range(0, n)]
    return temp


def large_samples(n, q):
    """returns a list"""

    temp = [randint(-q, q) for x in range(0, n)]
    return temp


class Rq(list):
    """
    notes: can't asign a list to a Rq object, eg a = [1,2,3]
    """
    n = 3
    q = 5

    def __init__(self, n=3, q=5, coeffs=[]):
        self.n = n
        self.q = q

        # addictive and multiplicative identity
        # self.zero = Rq(self.n, self.q, [0 for i in range(self.n)])
        # self.one = Rq(self.n, self.q, [0 for i in range(self.n)])
        # self.one[0] = 1


        if len(coeffs) > n:
            coeffs = mod_poly(coeffs, n)

        for i, j in enumerate(coeffs):
            if j >= q // 2 or j < 0:
                coeffs[i] = modmath(j, q)
        super(Rq, self).__init__(coeffs)

    def __setitem__(self, i, y):
        if y >= self.q or y < 0:
            y = modmath(y, self.q)
        super(Rq, self).__setitem__(i, y)

    def __len__(self):
        return self.n

    def __add__(self, y):
        # if len(self) != len(y):
        #     raise ValueError("different dimension")
        temp = Rq(self.n, self.q, self)

        for i, j in enumerate(y):
            temp[i] += j
            if temp[i] >= self.q / 2 or temp[i] < 0:
                temp[i] = modmath(temp[i], self.q)
        return Rq(self.n, self.q, temp)

    def __sub__(self, y):
        # if len(self) != len(y):
        #     raise ValueError("different dimension")
        temp = Rq(self.n, self.q, self)

        for i, j in enumerate(y):
            temp[i] -= j
            if temp[i] >= self.q / 2 or temp[i] < 0:
                temp[i] = modmath(temp[i], self.q)
        return Rq(self.n, self.q, temp)

    def __mul__(self, y):
        temp = []
        # if multiply with scalar
        if type(y) is int:
            for i, j in enumerate(self):
                temp.append(j * y)
                if temp[i] >= self.q / 2 or temp[i] < 0:
                    temp[i] = modmath(temp[i], self.q)
            return Rq(self.n, self.q, temp)

        # if multiply with another ring element
        for i, j in enumerate(self):
            temp.append(j)
        temp = mod_poly(poly_multiply(temp, y), self.n)
        for i, j in enumerate(temp):
            if temp[i] >= self.q / 2 or temp[i] < 0:
                temp[i] = modmath(temp[i], self.q)
        return Rq(self.n, self.q, temp)

    def __rmul__(self, n):
        temp = Rq(self.n, self.q, self)

        for i, j in enumerate(temp):
            temp[i] *= n
            if temp[i] >= self.q / 2 or temp[i] < 0:
                temp[i] = modmath(temp[i], self.q)

        return Rq(self.n, self.q, temp)

    def __pow__(self, power, modulo=None):
        ori = Rq(self.n, self.q, self)
        temp = Rq(self.n, self.q, self)
        if power == 0:
            return Rq(self.n, self.q, [1 for x in range(self.n)])

        for i in range(power - 1):
            temp = temp * ori
        return temp

    def rot_matrix(self):
        ret = []
        for i in range(len(self)):
            # first column stay the same
            if i == 0:
                ret.append(self)
                continue
            # next columns
            temp = []
            # first element, negate the last element
            # of previous column
            neg = modmath(-ret[i - 1][-1], self.q)
            temp.append(neg)
            # append the others
            for j in range(0, len(self) - 1):
                temp.append(ret[i - 1][j])

            ret.append(temp)
        return ret

    @staticmethod
    def add_matrix(a,b,q):
        ret = []
        for i, j in zip(a,b):
            temp = []
            for k,l in zip(i,j):
                temp.append(k+l)
            ret.append(temp)
        ring_ret = []
        for i in ret:
            temp = Rq(len(i), q, i)
            ring_ret.append(temp)
        return ring_ret

    @staticmethod
    def inner_product(a, b, q):
        # inner product of 2 vectors, result a
        # scalar in Zq
        temp = 0
        for i, j in zip(a, b):
            temp += i * j
        return modmath(temp, q)

    @staticmethod
    def vec_mult_matrix(a, A, q):
        # product of a vector and a matrix, result
        # a vector
        ret = []
        for i in A:
            temp = Rq.inner_product(a,i, q)
            ret.append(temp)
        return ret

    @staticmethod
    def random_samples(n, q):
        """return a ring element"""
        temp = large_samples(n, q)
        ret = Rq(n, q, temp)
        return ret


class BV(object):
    def __init__(self, n=3, q=40433, t=2, sigma=4):
        super(BV, self).__init__()
        self.n = n
        self.q = q
        self.t = t
        self.sigma = sigma
        self.zeros = Rq(self.n, self.q, [0 for i in range(self.n)])
        self.one = Rq(self.n, self.q, [0 for i in range(self.n)])
        self.one[0] = 1
        self.all_one=Rq(self.n, self.q, [1 for _ in range(self.n)])

    def small_samples(self):
        """return a Rq element"""
        temp = small_samples(self.n, self.sigma)
        ret = Rq(self.n, self.q, temp)
        return ret

    def large_samples(self):
        """return a ring element"""
        temp = large_samples(self.n, self.q)
        ret = Rq(self.n, self.q, temp)
        return ret

    def genkey(self):
        s = self.small_samples()
        p_1 = self.large_samples()
        e = self.small_samples()
        p_0 = self.zeros - (p_1 * s + self.t * e)

        #debug
        # print('original e %r'%e)
        # print('secret key s %r'%s)

        # for debug purpose only
        self.s = s
        self.pk = (p_0, p_1)

        #debug
        # print('original p_0 %r'%p_0)
        # print('original p_1 %r'%p_1)

        return (s, (p_0, p_1))

    def pad_ring(self, c1, c2):
        if len(c1) == len(c2):
            return
        if len(c1) > len(c2):
            no_zero = len(c1) - len(c2)
            for i in range(no_zero):
                c2.append(self.zeros)
        else:
            no_zero = len(c2) - len(c1)
            for i in range(no_zero):
                c1.append(self.zeros)
        return

    def enc(self, m, pk):
        u = self.small_samples()
        f = self.small_samples()
        g = self.small_samples()

        # debug
        # print('random u %r'%u)
        # print('random f %r'%f)
        # print('random g %r'%g)

        # compute a new sesssion key (a mask) from public
        # key and encrypt it with the new generated mask
        (c_0, c_1) = (pk[0] * u + self.t * g + m, pk[1] * u + self.t * f)

        #debug
        # print('original c_0 %r',c_0)
        # print('original c_1 %r',c_1)

        return [c_0, c_1]

    def dec(self, c, sk):
        len_c = len(c)
        s = []
        for i in range(len_c):
            if i == 0:
                s.append(self.one)
                continue
            s.append(sk ** i)
        temp = Rq(self.n, self.q, self.zeros)
        for i, j in zip(c, s):
            temp = temp + i * j

        # break here to see how large the error is
        for i, j in enumerate(temp):
            temp[i] = modmath(j, self.t)
        return temp

    def add(self, c1, c2):
        if len(c1) != len(c2):
            self.pad_ring(c1, c2)

        temp = []
        for i, j in zip(c1, c2):
            temp.append(i + j)

        # result ciphertext is a tuple of ring elements
        return temp

    def sub(self, c1, c2):
        if len(c1) != len(c2):
            self.pad_ring(c1, c2)

        temp = []
        for i, j in zip(c1, c2):
            temp.append(i - j)

        # result ciphertext is a tuple of ring elements
        return temp

    def mult(self, c1, c2):
        if len(c1) != len(c2):
            self.pad_ring(c1, c2)
        # degree of the cross product result
        N = 2 * (len(c1) - 1)
        temp = []
        # append the first item
        temp.append(c1[0] * c2[0])

        # append the mid item
        # collect the coeffs for each degree
        for i in range(1, N):
            coeff = Rq(self.n, self.q, self.zeros)
            for j in range(i, -1, -1):
                # sometime it might go pass the index
                try:
                    coeff = coeff + c1[j] * c2[i - j]
                except:
                    continue
            temp.append(coeff)

        # append the last item
        temp.append(c1[-1] * c2[-1])
        return temp

    def pack1(self, m):
        return m

    def pack2(self, m):
        temp = Rq(self.n, self.q)
        temp = m
        ret = []
        ret.append(temp[0])
        temp = self.zeros - temp

        for i, j in enumerate(reversed(temp)):
            if i == len(temp) - 1:
                break
            ret.append(j)

        return Rq(self.n, self.q, ret)

    def compute_hd(self, pm1, pm2):
        temp1 = [-1 for _ in range(self.n)]
        temp1[0] = 1
        c1 = Rq(n=self.n, q=self.t, coeffs=temp1)
        enc_c1 = self.enc(c1, self.pk)
        temp2 = [1 for _ in range(self.n)]
        c2 = Rq(n=self.n, q=self.t, coeffs=temp2)
        enc_c2 = self.enc(c2, self.pk)

        first_part = self.mult(pm1, enc_c1)
        sec_part = self.mult(pm2, enc_c2)
        third_part = self.mult(pm1, pm2)

        ret = self.add(first_part, sec_part)
        ret = self.sub(ret, third_part)
        ret = self.sub(ret, third_part)

        return ret

    # represent message m in unary, power instead of value
    def unary_encode(self, m):
        """represent message in terms of polynomial"""
        if m >= 2 * self.n:
            m = m % (2 * self.n)
        temp = [0 for _ in range(0, self.n)]
        if m < self.n:
            temp[m] = 1
        else:
            temp[m % self.n] = -1
        return Rq(self.n, self.q, temp)

    def unary_decode(self, m):
        """get back the message from the polynomial degree d, message space 2d"""


        for i in range(len(m)):
            if m[i] == 1:
                return i
            if m[i] == -1:
                return self.n + i

    def to_lwe(self, c):
        """convert bv ciphertext to lwe ciphertext
        by inner product with vector 1
        """
        rot_c0 = c[0].rot_matrix()
        l0 = Rq.vec_mult_matrix(self.all_one, rot_c0, self.q)[0]

        rot_c1 = c[1].rot_matrix()
        l1 = Rq.vec_mult_matrix(self.all_one, rot_c1, self.q)

        return [l0, l1]

    def decrypt_lwe(self, c, sk):
        """TODO: noise analysis when multiply with rot
        matrix
        """
        t = Rq.inner_product(c[1],sk, self.q) + c[0]
        # print('before denoise %r',t)
        t = modmath(t, self.q)
        t = modmath(t, self.t)
        return t
