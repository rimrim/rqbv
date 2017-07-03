from ctypes import cdll
lib = cdll.LoadLibrary('./libfoo.so')

class Foo(object):
    def __init__(self):
        self.obj = lib.Foo_new()

    def bar(self):
        lib.Foo_bar(self.obj)

    def test_ntt(self):
        lib.test_ntt(self.obj)

f = Foo()
f.bar()
f.test_ntt()

