from bv import poly_multiply, BV, Rq, modmath, \
    small_samples, large_samples, rot
from timer import Timer
from bitarray import bitarray
from math import log,ceil,floor

def bit_repr(x, d):
    """Represent number x using d bits"""
    bit_decomp = bin(x)[2:].zfill(d)[::-1]
    ret = bitarray(bit_decomp)
    return ret

n = 100
q = 2**100 - 3
sigma = 3
t = 2*n
l = floor(log(t, 2)) - 1

#Time to compute HD
bv = BV(n = n, t = t, q = q, sigma= sigma)
(sk, pk) = bv.genkey()
X = [1 for _ in range(n)]
Y = [0 for _ in range(n)]
for i,j in enumerate(Y):
    if i % 2 == 0 :
        Y[i] = 1
X = Rq(n = n, q = t, coeffs= X)
Y = Rq(n = n, q = t, coeffs= Y)
pm1 = bv.pack1(X)
pm2 = bv.pack2(Y)
T = bv.enc(pm1, pk)
Q = bv.enc(pm2, pk)

#Compute HD
with Timer() as ti:
    C_HD = bv.compute_hd(T, Q)
print('time for server to compute HD is %s ms '%ti.msecs)

# mask the HD
r = [5 for _ in range(n)]
r = Rq(n = n, q = t, coeffs=r)
EncR = bv.enc(r, pk)
with Timer() as ti:
    C_HDMask = bv.add(C_HD, EncR)
print('time for server to add masking is %s ms'%ti.msecs)

#decrypt HD Masked
with Timer() as ti:
    plain_hd = bv.dec(C_HDMask, sk)[0]
    print('plain hd is %s '%plain_hd)
print('time for client to decrypt HD is %s ms '%ti.msecs)

#bits encryption
bits = bit_repr(plain_hd, l).to01()
list_bits = []
for i,j in enumerate(bits):
    temp = [0 for _ in range(n)]
    temp[0] = i
    list_bits.append(Rq(n = n, q = t, coeffs=temp))
enc_b = []
with Timer() as ti:
    for i in list_bits:
        C_i = bv.enc(i, sk)
        enc_b.append(C_i)
print('time for client to encrypt %s ciphertext is %s ms'%(len(bits),ti.msecs))

#convert from Enc(b_i) to Enc(x^jb_i)
enc_b2 = []
with Timer() as ti:
    for i,j in enumerate(enc_b):
        c = bv.map_to_j(j, 2**i)
        enc_b2.append(c)
print('time for server to convert the ciphertexts from Enc(b) to Enc(x^jb) is %s ms'%ti.msecs)

