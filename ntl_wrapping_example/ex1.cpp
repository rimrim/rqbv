#include <NTL/ZZ.h>
#include <stdio.h>

using namespace std;
using namespace NTL;

NTL_CLIENT

int main()
{
    ZZ a, b, c;
    cout << "Enter a: " << "\n";
    cin >> a;

    cout << "Enter b: " << "\n";
    cin >> b;

    c = (a+1)*(b+1);

    cout << c << "\n";

    return 0;
}
