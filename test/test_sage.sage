# -*- mode: sage-shell:sage -*-
def test_sage_number_theory():
    # to construct a univariate ring, variable t
    R.<t> = PolynomialRing(QQ)
    # from here on can use t as symbolic variable
    f = t^5 + t^2 - 1
    print(factor(f))
    # to change coefficents to list
    print(list(f))
    # can do normal polynomial operations
    g = t^3
    print(f+g)
    print(f + g^2)
    h=(R.cyclotomic_polynomial(10))
    print(h)
    k = t^10 - 1
    print(k/h)

test_sage_number_theory()

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

# def eval_poly(f, n , q, var):
#     Z = GF(q)
#     d = var('d')
#     R = PolynomialRing(Z,'d')
#     S = R.quotient(d^n + 1, 'a')
#     a = S.gen()

#     t = f(x=x^var)
#     return S(t)

# example of ring init

# n = 16
# q = 257
# Z = GF(q)
# R = PolynomialRing(Z,'x')
# S = R.quotient(x^n + 1, 'a')
# a = S.gen()
# b = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# c = S(b)

# x = var('x')
# f = c[0] + c[1]*x + c[2]*x^2 + c[3]*x^3 + c[4]*x^4 + c[5]*x^5 + c[6]*x^6 + c[7]*x^7 + c[8]*x^8 + \
#     c[9]*x^9 + c[10]*x^10 + c[11]*x^11 + c[12]*x^12 + c[13]*x^13 + c[14]*x^14 + c[15]*x^15 + c[16]*x^16

# t = f(x=x^3)
# rot = S(t)
# print(rot)
# t = f(x=x^5)
# rot = S(t)
# print(rot)
# t = f(x=x^7)
# rot = S(t)
# print(rot)
# t = f(x=x^9)
# rot = S(t)
# print(rot)

# t = f(x=x^11)
# rot = S(t)
# print(rot)
# t = f(x=x^13)
# rot = S(t)
# print(rot)
# t = f(x=x^15)
# rot = S(t)
# print(rot)
# t = f(x=x^17)
# rot = S(t)
# print(rot)

# t = f(x=x^19)
# rot = S(t)
# print(rot)
# t = f(x=x^21)
# rot = S(t)
# print(rot)
# t = f(x=x^23)
# rot = S(t)
# print(rot)
# t = f(x=x^25)
# rot = S(t)
# print(rot)

# t = f(x=x^27)
# rot = S(t)
# print(rot)

# t = f(x=x^29)
# rot = S(t)
# print(rot)
# t = f(x=x^31)
# rot = S(t)
# print(rot)
# t = f(x=x^33)
# rot = S(t)
# print(rot)
# t = f(x=x^35)
# rot = S(t)
# print(rot)




# A = random_matrix(S,3,2)
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
