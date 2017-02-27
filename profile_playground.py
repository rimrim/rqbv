from bv import BV
from timer import Timer

bv = BV(n=20, q=2 ** 60, t=40, sigma=2)
(sk, pk) = bv.genkey()
m1 = bv.small_samples()
m2 = bv.small_samples()
m3 = bv.small_samples()
m4 = bv.small_samples()
m5 = bv.small_samples()
m6 = bv.small_samples()

c1 = bv.enc(m1, pk)
c2 = bv.enc(m2, pk)
c3 = bv.enc(m3, pk)
c4 = bv.enc(m4, pk)
c5 = bv.enc(m5, pk)
c6 = bv.enc(m6, pk)

c = [c1,c2,c3,c4,c5,c6]
with Timer() as ti:
    p = bv.bin_tree_mult(c)
print('time to bin mul %s ms'%ti.msecs)
plain = bv.dec(p, sk)
assert(m1 * m2 * m3 * m4 * m5 * m6 == plain)
