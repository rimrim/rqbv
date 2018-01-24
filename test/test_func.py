import unittest

from auth_protocol import AuthProtocol
from bv import poly_multiply, BV, BGV, small_samples, large_samples

from ntt import *
from mymath import *
from timer import Timer




class FunctionalTest(unittest.TestCase):
    def test_protocol4(self):
        print('This is protocol 4')
        # - Set up parameters
        #   - Biometrics parameters
        #     - n, power of 2
        #     - FAR, FRR
        n = 16
        far = 10**(-5)
        #   - Cryptosystem parameters
        #     - n, power of 2
        #     - q, prime, plaintext space, prime so that NTT can be done
        #     - \alpha q, small
        qbits = 8
        q = set_up_params(n, qbits)
        e_0 = 2

        # we need to derive some other parameters for NTT operations
        k = (q-1)//n
        g = find_generator(q-1, q) #generator
        omega = modmath(g**k, q) #n-th root of unity
        psi = square_root_mod(q, n) #2n-th root of unity

        # - Enrolment
        #   - extract template T
        T = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        T = Rq(n, q, T)
        #   - compute enc(T):
        #     - encode T with NTT(T)
        #  - note that this is special of special NTT, that is rotateable and operations done in x^n + 1
        #  - compute NTT_plus(T)
        T_hat = transform_plus(T)
        #  - rearrange the result to get NTT(T), let's call this T_bar
        #     - encrypt T_bar, return enc(T_bar), send enc(T_bar) to the server
        # - Authentication
        #   - extract template Q
        #   - compute enc(Q)
        #     - encode Q with NTT(Q)
        #  - compute NTT_plus(Q)
        #  - rearrange the result to get NTT(Q), let's call this Q_bar
        #     - encrypt Q_bar, return enc(Q_bar), send enc(Q_bar) to the server
        #   - Run ZKP to prove Q binary
        #   - Run ZKP to prove Q is valid ciphertext
        #   - Compute enc(HD)
        #     - compute enc(T) XOR enc(Q) = enc(T) + enc(Q) - 2.enc(T).enc(Q) = enc(xor)
        #     - rotate and add enc(xor)
        #   - Sample random r and noise e_2
        #   - Add noise to enc(HD): enc(HDD,e_tot) = enc(HD,e_1) + enc(r, e_2)
        #   - Compile the garble circuit (GC) to compute HD < \tau, given r and HDD
        GC = AuthProtocol()
        #   - Send the enc(HDD), k_r_i and GC to client
        #   - Client decrypt enc(HDD) to get HDD
        #   - Client decompose HDD to its binary representation (h_0, h_1, ..., h_{l-1})
        #   - Client encrypt enc(h_0), enc(h_1), ..., enc(h_{n-1})
        #   - Client proves that he decrypts and decompose honestly and correctly
        #   - Client runs OT protocol: given enc(h_i), get back enc(k_h_i)
        #   - Client decrypts enc(k_h_i) and use that together with k_r_i as input to GC
        #   - Client sends GC output to server
        #   - Server outputs the authentication result

if __name__ == '__main__':
    unittest.main()
