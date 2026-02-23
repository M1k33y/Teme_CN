#ex1
import math
from math import copysign, pi, tan
import random
import time

def precizie_masina():
    m=0
    u=10**(-m)
    while 1+u!=1:
        m+=1
        u=10**(-m)
    m-=1
    return 10**(-m)

print ("Precizia masina este:", precizie_masina())

#ex2

u=precizie_masina()
x=1.0
y=u/10
z=u/10

def verificare_adunare(x,y,z):
    rez1=(x+y)+z
    rez2=x+(y+z)

    print("(x+y)+z =", rez1)
    print("x+(y+z) =", rez2)
    
verificare_adunare(x,y,z)

x=u
y=u+1
z=u+1

def verificare_inmultire(x,y,z):
    rez1=(x*y)*z
    rez2=x*(y*z)
    print("(x*y)*z =", rez1)
    print("x*(y*z) =", rez2)

verificare_inmultire(x,y,z)

#ex3

#epsilon = diferenta in valoare absoluta intre fi si fi-1 >epsilon dat (practic precizia)
#x e elem din tan x 

def metoda_fractii(x, eps=1e-12):
    mic = 1e-12
    
    b0 = 0.0
    f = b0
    if f == 0:
        f = mic
    
    C = f
    D = 0.0
    
    j = 1
    
    while True:
        if j == 1:
            a = x
            b = 1.0
        else:
            a = -x*x
            b = 2*j - 1
        
        D = b + a * D
        if D == 0:
            D = mic
        
        C = b + a / C
        if C == 0:
            C = mic
        
        D = 1.0 / D
        delta = C * D
        f = f * delta #fi=Cj*Dj*fi-1
        
        if abs(delta - 1.0) < eps: #conditia de oprire a whileului
            break 
        
        j += 1
    
    return f

    
c1 = 0.33333333333333333
c2=0.133333333333333333
c3=0.053968253968254
c4=0.0218694885361552

# c1=1/3
# c2=2/15
# c3=17/315
# c4=62/2835

def polinom (x):
    return c1*x+c2*x+c3*x**2+c4*x**3

def metoda_polinom(x):
    return x + polinom(x)*(x**3)

def metoda_polinom_optim(x):
    
    sign = 1.0
    #am nevoie de sign sa schimb semnu tan
    # tan(x)=-tan(-x)
    if x < 0:
        sign = -1.0
        x = -x

    use = False

    if x > pi/4:
        x = pi/2 - x
        use = True

    #schema lui horner
    x2 = x * x
    x3 = x2 * x

    p = c4
    p = c3 + x2 * p
    p = c2 + x2 * p
    p = c1 + x2 * p

    t = x + x3 * p

    if use:
        t = 1.0 / t 
    #tg(x) = 1/tg(pi/2-x)

    return sign * t

x = float(input("Scrie x: "))
print(tan(x)," ", metoda_fractii(x)," ", metoda_polinom(x), " ", metoda_polinom_optim(x))

#rezolvare efectiva

N= 10000

nr=[random.uniform(-pi/2,pi/2) for i in range(N)]

start_time = time.time()
diff1 =[]
for x in nr:
    aproximare=metoda_fractii(x)
    real=tan(x)
    diff1.append(abs(aproximare-real))
end_time = time.time()

time1=end_time - start_time
print("Timpul pentru metoda fractii este:", time1)


start_time = time.time()
diff2 =[]
for x in nr:
    aproximare=metoda_polinom_optim(x)
    real=tan(x)
    diff2.append(abs(aproximare-real))
end_time = time.time()

time2=end_time - start_time
print("Timpul pentru metoda polinom este:", time2)

m1=sum(diff1)/N
m2=sum(diff2)/N
print("Media erorilor pentru metoda fractii este:", m1)
print("Media erorilor pentru metoda polinom este:", m2)
