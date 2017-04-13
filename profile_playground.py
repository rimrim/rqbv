import cProfile
from cProfile import Profile
from pstats import Stats

from bv import BV, large_samples, Rq
from timer import Timer

q = 2**62 + 1
n = 1000
n1 = Rq(n = n, q = q, coeffs=large_samples(n,q))
n2 = Rq(n = n, q = 2**32+1, coeffs=large_samples(n,q))
with Timer() as ti:
    m = n1*n2
print('time for 1 multiplication for 32 bit Q is %s ms'%ti.msecs)
n1 = Rq(n = n, q = 2**64-3, coeffs=large_samples(n,q))
n2 = Rq(n = n, q = 2**64-3, coeffs=large_samples(n,q))
with Timer() as ti:
    m = n1*n2
print('time multiplication for 64 bit Q is %s ms'%ti.msecs)
n1 = Rq(n = n, q = 2**128+3, coeffs=large_samples(n,q))
n2 = Rq(n = n, q = 2**128+3, coeffs=large_samples(n,q))
with Timer() as ti:
    m = n1*n2
print('time multiplication for 128 bit Q is %s ms'%ti.msecs)
with Timer() as ti:
    m = n1+n2
print('time addition for 128 bit Q is %s ms'%ti.msecs)


bv = BV(n = n, q = 2**128+3, t = n, sigma=3)
(sk,pk) = bv.genkey()
with Timer() as ti:
    temp = pk[0]*n2
print('time to test %s'%ti.msecs)


# def test_enc():
#     enc_n = bv.enc(n1, pk)
# profiler = Profile()
# profiler.runcall(test_enc)
# stats = Stats(profiler)
# # stats.print_stats()
# stats.print_callers()
# # cProfile.runctx('test_enc()',None,locals(),filename='./test_enc')
