b = var('b')
a = var('a')
s = var('s')
e = var('e')
r1 = var('r1')
r2 = var('r2')
m1 = var('m1')
m2 = var('m2')
t = var('t')

b = a*s + t*e

c1 = b*r1 + m1
c2 = -a*r1
c3 = b*r2 + m2
c4 = -a*r2

dec = c1*c3 + s*(c1*c4 + c2*c3) + (s^2)*(c2*c4)

print(dec.expand().collect(m1*m2))

