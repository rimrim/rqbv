# print('hello world')
# print(gcd(22776,2452))
# The schnorr protocol

# init the protocol with a client and a server
# client = Client()
# server = Server()
# Protocol = Protocol(Client, Server)

# init the protocol with g, p
n = 201
CN = CyclicPermutationGroup(n)
a = CN.random_element()
H = CN.subgroup([a])
print(H)

# The client has a discret log secret and he wants to prove that he knows the secret

# client sends commit

# pro
