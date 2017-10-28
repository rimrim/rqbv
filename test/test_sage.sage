# -*- mode: sage-shell:sage -*-
def test( ):
    print("hello world")

def small_samples(n, sigma):
    """returns a list"""
    temp = [randint(0, sigma) for x in range(0, n)]
    return temp

def bit_samples(n):
    """returns a list"""
    temp = [randint(0, 1) for x in range(0, n)]
    return temp

# example of ring init
# n = 3
# q = random_prime(2**10)
# Z = GF(q)
# R = PolynomialRing(Z,'x')
# S = R.quotient(x^n + 1, 'a')
# a = S.gen()
# b = [1,2,3,4,5,6,7,8,9,0,1]
# c = S(b)
# print(c)
# # A = random_matrix(S,3,2)
# print(A)

def E_Setup(qbits, d, n = 1, sigma=3):
    ret = {}
    ret['qbits'] = qbits
    ret['q'] = random_prime(2**qbits)
    ret['d'] = d
    ret['N'] = ceil(log(qbits,2))
    ret['n'] = n
    ret['sigma'] = sigma
    Z = GF(ret['q'])
    S = PolynomialRing(Z,'x')
    R = S.quotient(x^d + 1, 'a')
    a = R.gen()
    ret['R'] = R
    return ret

def E_SecretKeyGen(params):
    s_0 = [0 for _ in range(params['d'])]
    s_0[0] = 1
    s_0 = params['R'](s_0)
    s_1 = bit_samples(params['d'])
    s_1 = params['R'](s_1)
    return Matrix([s_0, s_1])
# params = E_Setup(10,3)
# # print(params['q'])
# sk = E_SecretKeyGen(params)
# print(sk)

def small_noise(params):
    e = []
    for i in range(params['N']):
        temp = small_samples(params['d'],params['sigma'])
        temp = params['R'](temp)
        e.append(temp)
    e = Matrix(e)
    return e.transpose()
# print(small_noise(params))

def E_PublicKeyGen(params, sk):
    A_prime = random_matrix(params['R'],params['N'], params['n'])
    c = -A_prime
    for i in c:
        print(type(i))
    e = small_noise(params)
    b = A_prime*sk[0][1] + 2*e
    # a = Matrix(params['R'],[b,c])
# pub = E_PublicKeyGen(params,sk)
# print(params['q'])
# print(pub)



# # BGV system
# # assume that we are working with RLWE setting (n = 1)
# # note that in the paper, it use d to denote dimention of the lattice
# n = 1
# d = 10
# # this is the level
# L = 5
# # This is the original bit length of q_0
# muy = 10
# # setup
# params = [ {} for _ in range(L+1)]
# for j in reversed(range(L+1)):
#     params[j] = E_Setup((j+1)*muy, d)
# # print(params)

# # keygen
# s = [None for _ in range(L+1)]
# for j in reversed(range(L+1)):
#    s[j] = E_SecretKeyGen(params[j]) 
# # enc
# # dec
