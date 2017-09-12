import numpy
from bitarray import bitarray
from random import randint
from math import ceil, log

# note: it's hard to install scipy on windows
from scipy.fftpack import fft, ifft
from scipy import signal
from numpy import real, base_repr, convolve

class Gadget(object):
    """Flattening gadget utility""" 
    def __init__(self, base, length):
        self.base = base
        self.length = length

    def forward(self, a):
        ret = []
        if isinstance(a, Rq ):
            for i in a:
                if i < 0:
                    i = a.q + i
                ret.append(self.forward(i))
            return ret
            
        if isinstance(a, int):
            ret = base(a, self.base)
            if len(ret) > self.length:
                raise ValueError('base length too short')
            if len(ret) < self.length:
               ret.extend([0]*(self.length - len(ret)))
            return ret

        if isinstance(a, list):
            for i in a:
                ret.append(self.forward(i))
            return ret
    

    def backward(self, b):
        ret = 0
        ret_list = []
        for i,j in enumerate(b):
            if isinstance(j,int):
                ret += j*(self.base**i)
            if isinstance(j,list):
                ret_list.append(self.backward(j))
        if ret_list:
            return ret_list
        return ret

class Bitarray(bitarray):

    def __lshift__(self, count):
        return self[count:] + type(self)('0') * count

    def __rshift__(self, count):
        return type(self)('0') * count + self[:-count]

    @property
    def bytes(self):
        return self.tobytes()

def extract_list_ring(list):
    temp = []
    for i in list:
        for j in i:
            temp.append(j)
    return temp


def pow_base(ring, q, b = 2):
    """power base b one ring element to log_base(q) ring elements, large norm"""
    length = ceil(log(q,b))
    final = []
    for i in range(length):
        temp = (b**i)*ring
        final.append(temp)
    return final

def decomp(ring, q, b = 2):
    """Decompose one ring element to log_base(q) ring elements, smaller norm"""
    temp = list(ring)
    length = ceil(log(q,b))
    for (i,j) in enumerate(temp):
        if j < 0:
            temp[i] = q + j
    final = []
    for _ in range(length):
        final.append([])
    for i in temp:
        t = base(i, b)
        if len(t) < length:
            t.extend([0]*(length - len(t)))
        for j,k in enumerate(t):
            final[j].append(k)
    return final
        
    

def base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits

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
    # return list(int(round(_) for _ in signal.convolve(a,b,'full')))
    # return numpy.polymul(a,b)


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

    def evaluate_at(self,x):
        temp = 0
        for i,j in enumerate(self):
            temp = temp + j*(x**i)
        return modmath(temp,self.q)

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

    @staticmethod
    def hd_plain(a, b):
        # work on bit string only
        temp = 0
        for (i,j) in zip(a,b):
            if i != j:
                temp += 1
        return temp

    @staticmethod
    def matrix_mul_ring(A,x):
        y = []
        for a in A:
            y.append(a*x)
        return y

    @staticmethod
    def perm_eval(pi, x):
        temp = []
        for i in pi:
            temp.append(x[i])
        return temp

    def encode(self):
        """Represent the positive integer 'x' using 'd' bits in a byte string.

        If 'd' is not a multiple of 8, 0-valued bits are prepended to
        make up a complete byte.

        ValueError is raised if the number of bits is insufficient to represent
        the value."""
        temp = Rq(self.n, self.q, self)
        d = self.q
        ret = b''
        for x in temp:
            if x < 0:
                x = x + self.q
            bit_decomp = bin(x)[2:]
            if len(bit_decomp) > d:
                raise ValueError(
                        "{:d} is too large to represent in {:d} bits.".format(x, d))
            # Increase d to the next multiple of 8.
            # This is needed as Bitarray.tobytes() pads an incomplete byte on
            # the right hand side.
            rem = d % 8
            if rem != 0:
                d = d + 8 - rem
            # Create a string of '01' characters, padded with leading '0'.
            bit_decomp = bit_decomp.zfill(d)
            # Convert the string to bytes.
            ret += Bitarray(bit_decomp).tobytes()
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

        temp1 = [-1 for _ in range(self.n)]
        temp1[0] = 1
        c1 = Rq(n=self.n, q=self.t, coeffs=temp1)
        self.enc_c1 = self.enc(c1, self.pk)
        self.enc_c2 = self.enc(self.all_one, self.pk)

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
        c_0 = pk[0] * u + self.t * g + m
        c_1 = pk[1] * u + self.t * f


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

        # print(self.q/max(temp))
        self.last_noise = max(temp)
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
        # if len(c1) != len(c2):
        #     self.pad_ring(c1, c2)
        # degree of the cross product result
        N = len(c1) + len(c2) -2
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


        first_part = self.mult(pm1, self.enc_c1)
        sec_part = self.mult(pm2, self.enc_c2)
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

    def to_lwe_generic(self, c):
        ret = []
        for i in c:
            rot = i.rot_matrix()
            temp = Rq.vec_mult_matrix(self.all_one, rot, self.q)
            ret.append(temp)
        ret[0] = ret[0][0]
        return ret

    def decrypt_lwe_generic(self, c, sk):
        """when c has more than 2 terms
        """
        temp = c[0]
        for i in range(1,len(c)):
            temp = temp + Rq.inner_product(c[i],sk**i, self.q)
        # t = Rq.inner_product(c[2],sk**2, self.q) + Rq.inner_product(c[1],sk**1, self.q) + c[0]
        # print('before denoise %r',t)
        t = modmath(temp, self.q)
        self.last_noise = t
        t = modmath(temp, self.t)
        return t

    def map_to_j_neg(self, c, j):
        c11 = [0 for _ in range(self.n)]
        c11[0] = 1
        c11[self.n - j] = 1
        c11 = Rq(self.n, self.t, c11)
        # c1 = self.enc(c11,self.pk)
        c2 = [self.one,self.zeros]
        c1 = [c11, self.zeros]
        mult = self.mult(c,c1)
        add = self.add(mult, c2)
        return add

    def map_to_j(self,c, j):
        c11 = [0 for _ in range(self.n)]
        c11[0] = -1
        c11[j] = 1
        c11 = Rq(self.n, self.t, c11)
        c2 = [self.one, self.zeros]
        c1 = [c11, self.zeros]
        # c1 = self.enc(c11,self.pk)
        mult = self.mult(c, c1)
        return self.add(mult, c2)

    def bin_tree_mult(self, c_list):
        temp = c_list
        l = len(temp)
        while (l != 1):
            for i,j in enumerate(temp):
                if i != (l-1-i):
                    temp[i] = self.mult(temp[i],temp[l-1-i])
                    temp.remove(temp[l-1-i])
            l = len(temp)
        return temp[0]

    def bit_repr(self, x, d):
        """Represent number x using d bits"""
        bit_decomp = bin(x)[2:].zfill(d)[::-1]
        ret = bitarray(bit_decomp)
        return ret
