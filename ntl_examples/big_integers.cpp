#include <NTL/ZZ.h>
using namespace std;

namespace test{
    using namespace NTL;

    NTL::ZZ test_add_int(const NTL::ZZ& a, const NTL::ZZ& b){
        NTL::ZZ c;
        c = (a+1)*(b+1);
        return c;
    }

    //note that we have SqrMod, MulMod, etc built in
    NTL::ZZ PowerMod(const NTL::ZZ& a, const NTL::ZZ& e, const NTL::ZZ& n){
        if (e == 0) {
            return NTL::ZZ(1);
        }

        //Note the helper Numbit
        long k = NumBits(e);
        NTL::ZZ res;
        res = 1;

        for (long i = k-1 ; i >= 0; i--) {
            res = (res*res) % n;
            //note the bit helper
            if (bit(e, i) == 1){
                res = (res*a) % n;
            }
        }

        if (e < 0) {
           return InvMod(res, n); 
        }
        else{
            return res; 
        }
    }

    long witness(const ZZ& n, const ZZ& x)
    {
        ZZ m, y, z;
        long j, k;
        if (x == 0) return 0;

        k = 1;
        m = n/2;
        while (m %2 == 0){
            k++;
            m /= 2;
        }

        z = test::PowerMod(x, m, n);
        if (z == 1) return 0;

        j = 0;
        do {
            y = z;
            z = (y*y)%n;
            j++;
        } while (j < k && z != 1);

        return z !=1 || y != n-1;
    }

    long PrimeTest(const ZZ& n, long t)
    {
        if (n <= 1) return 0;
        PrimeSeq s;
        long p;
        p = s.next();
        while (p && p < 2000){
            if ((n %p) == 0) return (n == p) ; 
            p = s.next();
        }

        ZZ x;
        long i;
        for (i = 0; i < t; i++) {
            //helper: random number between 0 and n-t
            x = RandomBnd(n);
            if (test::witness(n, x)) return 0;
        }
    return 1;
    }

}

int main(){
    NTL::ZZ a, b, c, n;
    cout << "n: ";
    cin >> n;
    if (test::PrimeTest(n ,10)) {
       cout << n << " is probably prime\n"; 
    }
    else {
        cout << n <<" is composite\n";
    }
    //cin >> b;
    //cin >> c;
    //NTL::ZZ d;
    //d = test::PowerMod(a,b,c);
    //cout << d << "\n";
    //cout << "size of long is %d" << sizeof(long) << "\n";
    return 0;
}

