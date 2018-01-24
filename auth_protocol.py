import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
backend = default_backend()
iv = os.urandom(16)

# GCM mode
def encrypt(key, plaintext, associated_data=b''):
    # Generate a random 96-bit IV.
    iv = os.urandom(12)

    # Construct an AES-GCM Cipher object with the given key and a
    # randomly generated IV.
    encryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
        backend=default_backend()
    ).encryptor()

    # associated_data will be authenticated but not encrypted,
    # it must also be passed in on decryption.
    encryptor.authenticate_additional_data(associated_data)

    # Encrypt the plaintext and get the associated ciphertext.
    # GCM does not require padding.
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()

    return (iv, ciphertext, encryptor.tag)

def decrypt(key, associated_data, iv, ciphertext, tag):
    # Construct a Cipher object, with the key, iv, and additionally the
    # GCM tag used for authenticating the message.
    decryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv, tag),
        backend=default_backend()
    ).decryptor()

    # We put associated_data back in or the tag will fail to verify
    # when we finalize the decryptor.
    decryptor.authenticate_additional_data(associated_data)

    # Decryption gets us the authenticated plaintext.
    # If the tag does not match an InvalidTag exception will be raised.
    return decryptor.update(ciphertext) + decryptor.finalize()

# AES CBC used in garbled circuit
def enc(k , m):
    # cipher = Cipher(algorithms.AES(k), modes.CBC(iv), backend=backend)
    # encryptor = cipher.encryptor()
    # ct = encryptor.update(m) + encryptor.finalize()
    # return ct
    (iv,ciphertext,tag) = encrypt(k,m)
    return iv + ciphertext + tag

def dec(k, c):
    # cipher = Cipher(algorithms.AES(k), modes.CBC(iv), backend=backend)
    # decryptor = cipher.decryptor()
    # m = decryptor.update(c) + decryptor.finalize()
    # return m
    iv = c[0:12]
    ciphertext = c[12:-16]
    tag = c[-16:]
    return decrypt(k, b'',iv, ciphertext, tag)


class GarbleCircuit(object):
    def __init__(self):
        pass


    def generate_random_keys_for_circuit(self, l):
        self.substractors = []
        self.comparators = []
        self.xnors = []
        self.ands = []
        self.ors = []

        # assign random keys first, reconnect later
        for i in range(l):
            subs = Substractor()
            subs.assign_random_keys()
            self.substractors.append(subs)

            comp = Comparator()
            comp.assign_random_keys()
            self.comparators.append(comp)

            xnor = GateXNOr()
            xnor.assign_random_keys()
            self.xnors.append(xnor)

            ands = GateAnd()
            ands.assign_random_keys()
            self.ands.append(ands)

            ors = GateOr()
            ors.assign_random_keys()
            self.ors.append(ors)

    def generate_gc(self, l):
        #connect the keys according to the circuit
        for i in range(l-1,-1,-1):
            if i > 0:
                self.substractors[i-1].garbled_keys_in[2] = self.substractors[i].garbled_keys_out[1]

            self.comparators[i].garbled_keys_in[0] = self.substractors[i].garbled_keys_out[0]

            if i < l - 1:
                self.xnors[i].garbled_keys_in[0] = self.substractors[i].garbled_keys_out[0]
                self.xnors[i].garbled_keys_in[1] = self.comparators[i].garbled_keys_in[1]

                if i == l - 2:
                    self.ands[i].garbled_keys_in[0] = self.comparators[i+1].garbled_keys_out[0]
                else:
                    self.ands[i].garbled_keys_in[0] = self.ors[i+1].garbled_keys_out[0]
                self.ands[i].garbled_keys_in[1] = self.xnors[i].garbled_keys_out[0]

                self.ors[i].garbled_keys_in[0] = self.ands[i].garbled_keys_out[0]
                self.ors[i].garbled_keys_in[1] = self.comparators[i].garbled_keys_out[0]

        # produce the ciphertexts
        for i in range(l):
            self.garble(self.substractors[i])
            self.garble(self.comparators[i])
            self.garble(self.xnors[i])
            self.garble(self.ands[i])
            self.garble(self.ors[i])

    def eval_garbled_circuit(self, k_hd, k_r, k_tau):
        if len(k_hd) != len(k_r) or len(k_r) != len(k_tau) or len(k_hd) != len(k_tau):
            raise IndexError('hd,r,tau should be in same bitstring length')
        l = len(k_hd)

        for i in range(l-1,-1,-1):
            self.substractors[i].garbled_keys_in[0] = k_hd[i]
            self.substractors[i].garbled_keys_in[1] = k_r[i]
            if i == l - 1:
                self.substractors[i].garbled_keys_in[2] = self.substractors[i].garbled_keys_in[2][0]

            (subs_key0, subs_key_1) = self.substractors[i].eval_keys(k_hd[i], k_r[i], self.substractors[i].garbled_keys_in[2])

            if i > 0:
                self.substractors[i-1].garbled_keys_in[2] = subs_key_1

            # self.comparators[i].garbled_keys_in[0] = subs_key0
            # self.comparators[i].garbled_keys_in[1] = k_tau[i]
            self.comparators[i].garbled_key_out = self.comparators[i].eval_keys(subs_key0,k_tau[i])

            if i < l - 1:
                # self.xnors[i].garbled_keys_in[0] = subs_key0
                # self.xnors[i].garbled_keys_in[1] = k_tau[i]
                self.xnors[i].garbled_key_out = self.xnors[i].eval_keys(subs_key0, k_tau[i])

                if i == l - 2:
                    and_0 = self.comparators[i+1].garbled_key_out
                else:
                    and_0 = self.ors[i+1].garbled_key_out
                and_1 = self.xnors[i].garbled_key_out
                self.ands[i].garbled_key_out = self.ands[i].eval_keys(and_0, and_1)

                or_0 = self.ands[i].garbled_key_out
                or_1 = self.comparators[i].garbled_key_out
                self.ors[i].garbled_key_out = self.ors[i].eval_keys(or_0, or_1)

        return self.ors[0].garbled_key_out

    def garble(self, gate):
        # provided the keys of the wire, this function outputs the ciphertext table of a gate
        if gate.type == 'general':
            for i in (0,1):
                for j in (0,1):
                    gate.input[0] = i
                    gate.input[1] = j
                    gate.eval()
                    inner = enc(gate.garbled_keys_in[0][i],gate.garbled_keys_out[0][gate.output[0]])
                    outter = enc(gate.garbled_keys_in[1][j],inner)
                    gate.garbled_ciphertexts.append(outter)

        if gate.type == 'substractor':
            for i in (0,1):
                for j in (0,1):
                    for l in (0,1):
                        gate.input[0] = i
                        gate.input[1] = j
                        gate.input[2] = l
                        gate.eval()
                        c0 = enc(gate.garbled_keys_in[2][l],
                                enc(gate.garbled_keys_in[1][j],
                                    enc(gate.garbled_keys_in[0][i],gate.garbled_keys_out[0][gate.output[0]])))

                        c1 = enc(gate.garbled_keys_in[2][l],
                                 enc(gate.garbled_keys_in[1][j],
                                     enc(gate.garbled_keys_in[0][i],gate.garbled_keys_out[1][gate.output[1]])))
                        gate.garbled_ciphertexts.append((c0,c1))


class GateSimple(object):
    # 2 input 1 output gates
    def __init__(self):
        self.input = [0 for _ in range(2)]
        self.output = [0 for _ in range(1)]
        self.garbled_keys_in = ['' for _ in range(2)]
        self.garbled_keys_out = ['']
        self.garbled_ciphertexts = []
        self.type = 'general'

    def assign_random_keys(self):
        for i in range(len(self.input)):
            key0 = os.urandom(16)
            key1 = os.urandom(16)
            self.garbled_keys_in[i] = (key0, key1)
        for i in range(len(self.output)):
            key0 = os.urandom(16)
            key1 = os.urandom(16)
            self.garbled_keys_out[i] = (key0, key1)

    def eval_keys(self, k1, k2):
        # output a key given 2 input keys and the ciphertexts table built already
        if len(self.garbled_ciphertexts) == 0:
            raise AttributeError("garbled ciphertexts not built yet")
        for i in self.garbled_ciphertexts:
            try:
                k_out = dec(k1,dec(k2,i))
                return k_out
            except:
                continue
        raise ValueError('input key(s) not correct')

class GateXNOr(GateSimple):
    def __init__(self):
        super().__init__()

    def eval(self):
        if self.input[0] == 0 and self.input[1] == 0:
            self.output[0] = 1
        if self.input[0] == 0 and self.input[1] == 1:
            self.output[0] = 0
        if self.input[0] == 1 and self.input[1] == 0:
            self.output[0] = 0
        if self.input[0] == 1 and self.input[1] == 1:
            self.output[0] = 1

class GateOr(GateSimple):
    def __init__(self):
        super().__init__()

    def eval(self):
        if self.input[0] == 0 and self.input[1] == 0:
            self.output[0] = 0
        if self.input[0] == 0 and self.input[1] == 1:
            self.output[0] = 1
        if self.input[0] == 1 and self.input[1] == 0:
            self.output[0] = 1
        if self.input[0] == 1 and self.input[1] == 1:
            self.output[0] = 1

class GateAnd(GateSimple):
    def __init__(self):
        super().__init__()

    def eval(self):
        if self.input[0] == 0 and self.input[1] == 0:
            self.output[0] = 0
        if self.input[0] == 0 and self.input[1] == 1:
            self.output[0] = 0
        if self.input[0] == 1 and self.input[1] == 0:
            self.output[0] = 0
        if self.input[0] == 1 and self.input[1] == 1:
            self.output[0] = 1


class Comparator(GateSimple):
    def __init__(self):
        super().__init__()

    def eval(self):
        if self.input[0] == 0 and self.input[1] == 0:
            self.output[0] = 0
        if self.input[0] == 0 and self.input[1] == 1:
            self.output[0] = 0
        if self.input[0] == 1 and self.input[1] == 0:
            self.output[0] = 1
        if self.input[0] == 1 and self.input[1] == 1:
            self.output[0] = 0

class Substractor(GateSimple):
    def __init__(self):
        self.input = [0 for _ in range(3)]
        self.output = [0 for _ in range(2)]
        self.garbled_keys_in = ['' for _ in range(3)]
        self.garbled_keys_out = ['' for _ in range(2)]
        self.garbled_ciphertexts = []
        self.type = 'substractor'

    def eval(self):
        if self.input[0] == 0 and self.input[1] == 0 and self.input[2] == 0:
            self.output[0] = 0
            self.output[1] = 0
        if self.input[0] == 0 and self.input[1] == 0 and self.input[2] == 1:
            self.output[0] = 1
            self.output[1] = 1
        if self.input[0] == 0 and self.input[1] == 1 and self.input[2] == 0:
            self.output[0] = 1
            self.output[1] = 1
        if self.input[0] == 0 and self.input[1] == 1 and self.input[2] == 1:
            self.output[0] = 0
            self.output[1] = 1
        if self.input[0] == 1 and self.input[1] == 0 and self.input[2] == 0:
            self.output[0] = 1
            self.output[1] = 0
        if self.input[0] == 1 and self.input[1] == 0 and self.input[2] == 1:
            self.output[0] = 0
            self.output[1] = 0
        if self.input[0] == 1 and self.input[1] == 1 and self.input[2] == 0:
            self.output[0] = 0
            self.output[1] = 0
        if self.input[0] == 1 and self.input[1] == 1 and self.input[2] == 1:
            self.output[0] = 1
            self.output[1] = 1

    def eval_keys(self, k1, k2, k3):
        # output 2 keys given 3 input keys and the ciphertexts table built already
        if len(self.garbled_ciphertexts) == 0:
            raise AttributeError("garbled ciphertexts not built yet")
        for i in self.garbled_ciphertexts:
            try:
                k_out = dec(k1,dec(k2,dec(k3,i[0])))
                k_out2 = dec(k1,dec(k2,dec(k3,i[1])))
                return k_out, k_out2
            except:
                continue
        raise ValueError('input key(s) not correct')

class AuthProtocol(object):
    def __init__(self):
        pass

    def generate_circuit(self, hd, r, tau):
        # this circuit does check hd - r > tau or not, even bits in string
        l = len(hd)
        if l != len(r) or l != len(tau):
            print('fix input lenghts, should be identical')
            return

        self.substractors = []
        self.comparators = []
        self.xnors = []
        self.ands = []
        self.ors = []

        for i in range(l):
            subs = Substractor()
            self.substractors.append(subs)

            comp = Comparator()
            self.comparators.append(comp)

            xnor = GateXNOr()
            self.xnors.append(xnor)

            ands = GateAnd()
            self.ands.append(ands)

            ors = GateOr()
            self.ors.append(ors)

        # connect the wires
        for i in range(l-1,-1,-1):
            self.substractors[i].input[0] = int(hd[i])
            self.substractors[i].input[1] = int(r[i])
            self.substractors[i].eval()
            if i > 0:
                self.substractors[i-1].input[2] = self.substractors[i].output[1]

            self.comparators[i].input[0] = self.substractors[i].output[0]
            self.comparators[i].input[1] = int(tau[i])
            self.comparators[i].eval()

            if i < l - 1:
                self.xnors[i].input[0] = self.substractors[i].output[0]
                self.xnors[i].input[1] = int(tau[i])
                self.xnors[i].eval()

                if i == l - 2:
                    self.ands[i].input[0] = self.comparators[i+1].output[0]
                else:
                    self.ands[i].input[0] = self.ors[i+1].output[0]
                self.ands[i].input[1] = self.xnors[i].output[0]
                self.ands[i].eval()

                self.ors[i].input[0] = self.ands[i].output[0]
                self.ors[i].input[1] = self.comparators[i].output[0]
                self.ors[i].eval()

        return self.ors[0].output[0]

    def print_circuit(self):
        print('number of substractors ' + str(len(self.substractors)))
        print('number of comparators ' + str(len(self.comparators)))



# class Comparator10Bits(object):
#     def __init__(self):
#         self.input = [0 for _ in range(20)]
#         self.output = [0 for _ in range(1)]
#         self.garbled_keys_in = ['' for _ in range(20)]
#         self.garbled_keys_out = ['']
#         self.garbled_ciphertexts = []
#         self.type = 'comparator10bits'
#
#     def eval(self):
#         # greater than
#         if self.input[0] < self.input[1]:
#             return 0
#         elif self.input[0] > self.input[1]:
#             return 1
#         else:
#             if self.input[2] < self.input[3]:
#                 return 0
#             elif self.input[2] > self.input[3]:
#                 return 1
#             else:
#                 if self.input[4] < self.input[5]:
#                     return 0
#                 elif self.input[4] > self.input[5]:
#                     return 1
#                 else:
#                     if self.input[6] < self.input[7]:
#                         return 0
#                     elif self.input[6] > self.input[7]:
#                         return 1
#                     else:
#                         if self.input[8] < self.input[9]:
#                             return 0
#                         elif self.input[8] > self.input[9]:
#                             return 1
#                         else:
#                             if self.input[10] < self.input[11]:
#                                 return 0
#                             elif self.input[10] > self.input[11]:
#                                 return 1
#                             else:
#                                 if self.input[12] < self.input[13]:
#                                     return 0
#                                 elif self.input[12] > self.input[13]:
#                                     return 1
#                                 else:
#                                     if self.input[14] < self.input[15]:
#                                         return 0
#                                     elif self.input[14] > self.input[15]:
#                                         return 1
#                                     else:
#                                         if self.input[16] < self.input[17]:
#                                             return 0
#                                         elif self.input[16] > self.input[17]:
#                                             return 1
#                                         else:
#                                             if self.input[18] < self.input[19]:
#                                                 return 0
#                                             elif self.input[18] > self.input[19]:
#                                                 return 1
#                                             else:
#                                                 return 1
