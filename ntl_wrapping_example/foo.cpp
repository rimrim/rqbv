#include <iostream>
#include <NTL/ZZ.h>

using namespace std;
using namespace NTL;

class Foo{
    public:
        void bar(){
            std::cout << "hello word" << std::endl;
        }

        void test_ntt(){
            ZZ a, b, c;
            cout << "Enter a: " << "\n";
            cin >> a;

            cout << "Enter b: " << "\n";
            cin >> b;

            c = (a+1)*(b+1);

            cout << c << "\n";
        }

};

extern "C"{
    Foo* Foo_new(){return new Foo();}
    void Foo_bar(Foo* foo){foo->bar();}
    void test_ntt(Foo* foo){foo->test_ntt();}

}
