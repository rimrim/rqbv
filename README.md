# rqbv
POC for ASCIP 2017 paper

Basic usage:

Ring operation:

    init a ring element: 

        r1 = Rq(n=3, q=5, [1,2,3])

        r2 = Rq(n=3, q=5, [1,1,1])

    ring operation:

        radd = r1 + r2 -> [2,-2,-1]

        rmul = r1*r2

BV cryptosystem:

    init:

        bv = BV(n=3,q=113,sigma=2)

    keygen:

        (sk,pk) = bv.genkey()

    Encrypt and Decrypt:

        m = Rq(n=3, q=5, [1,1,1])

        c = bv.enc(m, pk)

        p = bv.dec(c, sk)

    Homomorphic operation:

        m1 = Rq(n=3, q=5, [1,1,1])

        m2 = Rq(n=3, q=5, [1,1,1])

        c1 = bv.enc(m1,pk)

        c2 = bv.enc(m2,pk)

        c_add = bv.add(c1,c2)

        c_mult = bv.mult(c1,c2)

    Other operations can be found in the func_test and test_rq
        



 
