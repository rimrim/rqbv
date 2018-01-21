# legendre symbol (a|m)
# note: returns m-1 if a is a non-residue, instead of -1
from math import ceil, log
from bitarray import bitarray
from random import randint

from bv import modmath, Rq
from ntt import find_generator


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

def base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits

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

def pow_base(ring, q, b = 2):
    """power base b one ring element to log_base(q) ring elements, large norm"""
    length = ceil(log(q,b))
    final = []
    for i in range(length):
        temp = (b**i)*ring
        final.append(temp)
    return final

def extract_list_ring(list):
    temp = []
    for i in list:
        for j in i:
            temp.append(j)
    return temp

def set_up_params(n, qbits):
  if not is_power2(n):
    raise ValueError("n must be power of 2")
  q = next_prime(2**qbits)
  while (q-1) % (2*n) != 0:
    q = next_prime(q+1)
  return q

def legendre(a, m):
  return pow(a, (m-1) >> 1, m)

# strong probable prime
def is_sprp(n, b=2):
  d = n-1
  s = 0
  while d&1 == 0:
    s += 1
    d >>= 1

  x = pow(b, d, n)
  if x == 1 or x == n-1:
    return True

  for r in range(1, s):
    x = (x * x)%n
    if x == 1:
      return False
    elif x == n-1:
      return True

  return False

# lucas probable prime
# assumes D = 1 (mod 4), (D|n) = -1
def is_lucas_prp(n, D):
  P = 1
  Q = (1-D) >> 2

  # n+1 = 2**r*s where s is odd
  s = n+1
  r = 0
  while s&1 == 0:
    r += 1
    s >>= 1

  # calculate the bit reversal of (odd) s
  # e.g. 19 (10011) <=> 25 (11001)
  t = 0
  while s > 0:
    if s&1:
      t += 1
      s -= 1
    else:
      t <<= 1
      s >>= 1

  # use the same bit reversal process to calculate the sth Lucas number
  # keep track of q = Q**n as we go
  U = 0
  V = 2
  q = 1
  # mod_inv(2, n)
  inv_2 = (n+1) >> 1
  while t > 0:
    if t&1 == 1:
      # U, V of n+1
      U, V = ((U + V) * inv_2)%n, ((D*U + V) * inv_2)%n
      q = (q * Q)%n
      t -= 1
    else:
      # U, V of n*2
      U, V = (U * V)%n, (V * V - 2 * q)%n
      q = (q * q)%n
      t >>= 1

  # double s until we have the 2**r*sth Lucas number
  while r > 0:
      U, V = (U * V)%n, (V * V - 2 * q)%n
      q = (q * q)%n
      r -= 1

  # primality check
  # if n is prime, n divides the n+1st Lucas number, given the assumptions
  return U == 0

# primes less than 212
small_primes = set([
    2,  3,  5,  7, 11, 13, 17, 19, 23, 29,
   31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
   73, 79, 83, 89, 97,101,103,107,109,113,
  127,131,137,139,149,151,157,163,167,173,
  179,181,191,193,197,199,211])

# pre-calced sieve of eratosthenes for n = 2, 3, 5, 7
indices = [
    1, 11, 13, 17, 19, 23, 29, 31, 37, 41,
   43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
   89, 97,101,103,107,109,113,121,127,131,
  137,139,143,149,151,157,163,167,169,173,
  179,181,187,191,193,197,199,209]

# distances between sieve values
offsets = [
  10, 2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6,
   6, 2, 6, 4, 2, 6, 4, 6, 8, 4, 2, 4,
   2, 4, 8, 6, 4, 6, 2, 4, 6, 2, 6, 6,
   4, 2, 4, 6, 2, 6, 4, 2, 4, 2,10, 2]

max_int = 2147483647

# an 'almost certain' primality check
def is_prime(n):
  if n < 212:
    return n in small_primes

  for p in small_primes:
    if n%p == 0:
      return False

  # if n is a 32-bit integer, perform full trial division
  if n <= max_int:
    i = 211
    while i*i < n:
      for o in offsets:
        i += o
        if n%i == 0:
          return False
    return True

  # Baillie-PSW
  # this is technically a probabalistic test, but there are no known pseudoprimes
  if not is_sprp(n): return False
  a = 5
  s = 2
  while legendre(a, n) != n-1:
    s = -s
    a = s-a
  return is_lucas_prp(n, a)

# next prime strictly larger than n
def next_prime(n):
  if n < 2:
    return 2
  # first odd larger than n
  n = (n + 1) | 1
  if n < 212:
    while True:
      if n in small_primes:
        return n
      n += 2

  # find our position in the sieve rotation via binary search
  x = int(n%210)
  s = 0
  e = 47
  m = 24
  while m != e:
    if indices[m] < x:
      s = m
      m = (s + e + 1) >> 1
    else:
      e = m
      m = (s + e) >> 1

  i = int(n + (indices[m] - x))
  # adjust offsets
  offs = offsets[m:]+offsets[:m]
  while True:
    for o in offs:
      if is_prime(i):
        return i
      i += o

def square_root_mod(q, n):
  g = find_generator(q-1, q)
  t = g**((q - 1)/(2*n))
  return modmath(t, q)

def is_power2(num):
  return num != 0 and ((num & (num - 1)) == 0)
