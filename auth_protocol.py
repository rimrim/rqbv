import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
backend = default_backend()
iv = os.urandom(16)
def enc(k , m):
    cipher = Cipher(algorithms.AES(k), modes.CBC(iv), backend=backend)
    encryptor = cipher.encryptor()
    ct = encryptor.update(m) + encryptor.finalize()
    return ct

def dec(k, c):
    cipher = Cipher(algorithms.AES(k), modes.CBC(iv), backend=backend)
    decryptor = cipher.decryptor()
    m = decryptor.update(c) + decryptor.finalize()
    return m


class GarbleCircuit(object):
    def __init__(self):
        pass


    def garble(self, gate):
        for i in range(len(gate.input)):
            key0 = os.urandom(16)
            key1 = os.urandom(16)
            gate.garbled_keys_in[i] = (key0, key1)
        for i in range(len(gate.output)):
            key0 = os.urandom(16)
            key1 = os.urandom(16)
            gate.garbled_keys_out[i] = (key0, key1)

        if gate.type == 'general':
            for i in (0,1):
                for j in (0,1):
                    gate.input[0] = i
                    gate.input[1] = j
                    gate.eval()
                    c = enc(gate.garbled_keys_in[1][j],
                            enc(gate.garbled_keys_in[0][i],gate.garbled_keys_out[0][gate.output[0]]))
                    gate.garbled_ciphertexts.append(c)

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

class Substractor(object):
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


class AuthProtocol(object):
    def __init__(self):
        self.substractors = []
        self.comparators = []
        self.xor = []

    def generate_circuit(self, l = 10):
        for i in range(l):
            subs = Substractor()
            self.substractors.append(subs)

        for i in range(l):
            comp = Comparator()
            self.comparators.append(comp)

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
