# compiling instruction:
# g++ -c -fPIC foo.cpp -o foo.o -lntl
# g++ -shared -Wl,-soname,libfoo.so -o libfoo.so foo.o -lntl
# https://stackoverflow.com/questions/145270/calling-c-c-from-python

# build cryptolibrary: use brew link openssl to find out the -I and -L option,
# also link with -lssl and -lcrypto


from ctypes import cdll
lib = cdll.LoadLibrary('./libfoo.so')

class Foo(object):
    def __init__(self):
        self.obj = lib.Foo_new()

    def bar(self):
        lib.Foo_bar(self.obj)

    def test_ntt(self):
        lib.test_ntt(self.obj)

    def test_speed(self):
        lib.test_speed(self.obj)

    def test_enc(self):
        lib.test_enc(self.obj)

f = Foo()
f.bar()
f.test_ntt()
f.test_enc()
# f.test_speed()
