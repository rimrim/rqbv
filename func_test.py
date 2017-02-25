from bv import poly_multiply, BV, Rq, modmath, \
    small_samples, large_samples, rot
from timer import Timer

n = 100
q = 2**100 - 3
sigma = 100
t = 2*n

#Time to compute HD
bv = BV(n = n, t = t, q = q, sigma= sigma)
(sk, pk) = bv.genkey()
X = [1 for _ in range(n)]
Y = [0 for _ in range(n)]
for i,j in enumerate(Y):
    if i % 2 == 0:
        Y[i] = 1
X = Rq(n = n, q = t, coeffs= X)
Y = Rq(n = n, q = t, coeffs= Y)
pm1 = bv.pack1(X)
pm2 = bv.pack2(Y)
T = bv.enc(pm1, pk)
Q = bv.enc(pm2, pk)
with Timer() as t:
    C_HD = bv.compute_hd(T, Q)
print('time to compute HD is %s ms '%t.msecs)

with Timer() as t:
    plain_hd = bv.dec(C_HD, sk)
    print('plain hd is %s '%plain_hd[0])
print('time to decrypt HD is %s ms '%t.msecs)


