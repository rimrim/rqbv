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
q = 2**100
sigma = 1
t = 100
l = floor(log(t, 2)) - 1
#change threshold to check authentication result
tau = 20

#Time to compute HD
bv = BV(n = n, t = t, q = q, sigma= sigma)
(sk, pk) = bv.genkey()
X = [1 for _ in range(n)]
# Y = [1 for _ in range(n)]
# Y[0] = 0
Y = [0 for _ in range(n)]
for i,j in enumerate(Y):
    if i % 2 == 0 or i%3==0 or i% 5==0 or i%7==0 or i%11==0:
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
print('original HD is %s'%bv.dec(C_HD,sk)[0])

# mask the HD, compute -HD-r
r = [5 for _ in range(n)]
r = Rq(n = n, q = t, coeffs=r)
print('noise to add to HD is %s'%r[0])
EncR = bv.enc(r, pk)
with Timer() as ti:
    C_HDMask = bv.add(C_HD, EncR)
    C_HDMask = bv.sub([bv.zeros,bv.zeros],C_HDMask)
print('time for server to add masking is %s ms'%ti.msecs)

#decrypt HD Masked
with Timer() as ti:
    plain_hd = bv.dec(C_HDMask, sk)[0]

print('time for client to decrypt HD is %s ms '%ti.msecs)
print('plain (- hd masked) is %s '%plain_hd)
plain_hd = -plain_hd
#bits encryption
bits = bit_repr(plain_hd, l).to01()
print('bit representation of (-HDMask) is -%s'%bits)
list_bits = []
for i,j in enumerate(bits):
    temp = [0 for _ in range(n)]
    temp[0] = -int(bits[i])
    list_bits.append(Rq(n = n, q = t, coeffs=temp))
enc_b = []
with Timer() as ti:
    for i in list_bits:
        C_i = bv.enc(i, pk)
        enc_b.append(C_i)
print('time for client to encrypt %s ciphertext is %s ms'%(len(bits),ti.msecs))
#
#convert from Enc(-b_i) to Enc(x^-jb_i)
enc_b2 = []
with Timer() as ti:
    for i,j in enumerate(enc_b):
        c = bv.map_to_j_neg(j, 2**i)
        enc_b2.append(c)
print('time for server to convert the ciphertexts from Enc(-b) to Enc(x^-jb) is %s ms'%ti.msecs)

#Compute x^-HD by homomorphic multiplications
with Timer() as ti:
    hdm = bv.bin_tree_mult(enc_b2)
print('time for server to compute x^-HDMask is %s ms'%ti.msecs)

#Compute x^2l + tau + x^-HDM + x^r
term1 = bv.unary_encode(2**l + tau)
term3 = bv.unary_encode(r[0])
with Timer() as ti:
    enc_term1 = bv.enc(term1, pk)
    enc_term2 = hdm
    enc_term3 = bv.enc(term3, pk)
    mult_term = bv.mult(enc_term2, enc_term3)
    mult_term2 = bv.mult(mult_term, enc_term1)
print('time for server to compute Enc(2^l + t - HD) is %s'%ti.msecs)


with Timer() as ti:
    c_msb = bv.to_lwe_generic(mult_term2)
print('time for server to compute Enc(MSB) is %s ms'%ti.msecs)

with Timer() as ti:
    msb = bv.decrypt_lwe_generic(c_msb, sk)
print('time for client to decrypt MSB is %s ms'%ti.msecs)

print(bit_repr(bv.unary_decode(bv.dec(mult_term2,sk)),l+1))


#
# print(bit_repr(comp, 7))

#
# #Time to compute MSB
# r = bv.unary_encode(r[0])
# Enc_r = bv.enc(r, pk)
# sub1 = bv.sub(Enc_r, hdm)
# print(bv.dec(sub1,sk))
