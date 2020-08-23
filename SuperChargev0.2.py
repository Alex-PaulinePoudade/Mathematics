########################################################################
# SuperCharge GNU3
# menu pour calculatrice Numworks v 0.2
# programmé par Alex-Pauline Poudade (AlexPauline.Poudade@lfree.fr) 18/06/2020

import math
from fractions import *


def fregyinf(f):
    d=2
    while Fraction(1,d)>=f:
        d=d+1
    return Fraction(1,d)
    
def egyptianFraction(Num,Den):
    l=[]
    f=Fraction(Num,Den)
    while f.numerator!=1:
        l.append(fregyinf(f))
        f=f-fregyinf(f)
    l.append(f)
    return l


def remove(string): 
    return string.replace(" ", "")

def pgcd (a,b):
    while b!=0:
        a,b=b,a%b
    return a

def num_after_comma(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

def nearToExact(n):
    m=num_after_comma(n)
    m=float(m)
    n=float(n)
    n2=10**m
    n1=n*n2

    a=n1
    b=n2
    inverse=0

    if n1>n2:
        a=n1
        b=n2
        inverse=0
    if n2>n1:
        a=n2
        b=n1
        inverse=1
 
    signe=0
    if n1*n2<=0:
       signe=1
    a=abs(a)
    b=abs(b)
    n1=abs(n1)
    n2=abs(n2)

    if n<1:
       if inverse==1:
          Num=n1/pgcd(a,b)
          Den=n2/pgcd(a,b)
       elif inverse==0:
            Num=n2/pgcd(a,b)
            Den=n1/pgcd(a,b)
    else:
       if inverse==0:
          Num=n1/pgcd(a,b)
          Den=n2/pgcd(a,b)
       elif inverse==1:
            Num=n2/pgcd(a,b)
            Den=n1/pgcd(a,b)

    if signe==1:
        Num=-Num

    if n1==n2:
       Num=n1
       Den=n2
    #print ( n," est égal au rationnel : ",int(Num),"/",int(Den))
    return int(Num),int(Den)
########################################################################
def prettyPrint(Expression,x= None):
    l=[]

    Expression=remove(Expression)
    Expression=Expression.replace(",", ".")
    #if x==None:
    # n=(evaluate(Expression))
    #else:
    # n=(evaluate(Expression,{ 'x': x  }))
    n=float(Expression)
    Num,Den=nearToExact(n)
    Num2=Num
    Den2=Den

    while abs(Num2>10000) or abs(Den2>10000):
     Num2=int(math.floor(Num2/2))
     Den2=int(math.floor(Den2/2))
#     print("num2=",Num2,Den2)

    myfrac=Fraction(Num2,Den2)
    Num3=myfrac.numerator
    Den3=myfrac.denominator

    Expression=Expression.replace(".", ",")

    car=x
    if x==None:
     car='x' 
    if abs(Num3)<1000 and Den3!=1:
     l=egyptianFraction(Num2,Den2)
     l=sorted(l)
     print("f(x=",car,")=", Expression,"=",n, "=","(",Num,"/",Den,")=~(",Num2,"/",Den2,")=","(",Num3,"/",Den3,")=",sep='',end='')
     for i in range(len(l)-1):
      print("(",Fraction(l[i].numerator),"/",Fraction(l[i].denominator),")+",sep='',end='')                   
     print("(",Fraction(l[i+1].numerator),"/",Fraction(l[i+1].denominator),")",sep='',end='')
    else:   
#     print("f(x=",car,")=", Expression,"=", n,"=(", Num,"/",Den,")",sep='',end='')
     print("f(x=",car,")=", Expression,"=",n, "=","(",Num,"/",Den,")=~(",Num2,"/",Den2,")=","(",Num3,"/",Den3,")",sep='',end='')

    print("",end='\n')
    return 0
########################################################################
import copy
from math import log  # version >= 3.5
from fractions import gcd
#import random
#import sys


t_list = [
        2,
        12,
        60,
        180,
        840,
        1260,
        1680,
        2520,
        5040,
        15120,
        55440,
        110880,
        720720,
        1441440,
        4324320,
        24504480,
        73513440
        ]



# primality test by trial division
def isprime_slow(n):
    if n<2:
        return False
    elif n==2 or n==3:
        return True
    elif n%2==0:
        return False
    else:
        i = 3
        while i*i <= n:
            if n%i == 0:
                return False
            i+=2
    return True        


# v_q(t): how many time is t divided by q
def v(q, t):
    ans = 0
    while(t % q == 0):
        ans +=1
        t//= q
    return ans


def prime_factorize(n):
    ret = []
    p=2
    while p*p <= n:
        if n%p==0:
            num = 0
            while n%p==0:
                num+=1
                n//= p
            ret.append((p,num))
        p+= 1

    if n!=1:
        ret.append((n,1))

    return ret


# calculate e(t)
def e(t):
    s = 1
    q_list = []
    for q in range(2, t+2):
        if t % (q-1) == 0 and isprime_slow(q):
            s *= q ** (1+v(q,t))
            q_list.append(q)
    return 2*s, q_list

# Jacobi sum
class JacobiSum(object):
    def __init__(self, p, k, q):
        self.p = p
        self.k = k
        self.q = q
        self.m = (p-1)*p**(k-1)
        self.pk = p**k
        self.coef = [0]*self.m

    # 1
    def one(self):
        self.coef[0] = 1
        for i in range(1,self.m):
            self.coef[i] = 0
        return self


    # product of JacobiSum
    # jac : JacobiSum
    def mul(self, jac):
        m = self.m
        pk = self.pk
        j_ret=JacobiSum(self.p, self.k, self.q)
        for i in range(m):
            for j in range(m):
                if (i+j)% pk < m:
                    j_ret.coef[(i+j)% pk] += self.coef[i] * jac.coef[j]
                else:
                    r = (i+j) % pk - self.p ** (self.k-1)                    
                    while r>=0:
                        j_ret.coef[r] -= self.coef[i] * jac.coef[j]
                        r-= self.p ** (self.k-1)

        return j_ret


    def __mul__(self, right):
        if type(right) is int:
            # product with integer
            j_ret=JacobiSum(self.p, self.k, self.q)
            for i in range(self.m):
                j_ret.coef[i] = self.coef[i] * right
            return j_ret
        else:
            # product with JacobiSum
            return self.mul(right)
    

    # power of JacobiSum（x-th power mod n）
    def modpow(self, x, n):
        j_ret=JacobiSum(self.p, self.k, self.q)
        j_ret.coef[0]=1
        j_a = copy.deepcopy(self)
        while x>0:
            if x%2==1:
                j_ret = (j_ret * j_a).mod(n)
            j_a = j_a*j_a
            j_a.mod(n)
            x //= 2
        return j_ret
    

    # applying "mod n" to coefficient of self
    def mod(self, n):
        for i in range(self.m):
            self.coef[i] %= n
        return self
    

    # operate sigma_x
    # verification for sigma_inv
    def sigma(self, x):
        m = self.m
        pk = self.pk
        j_ret=JacobiSum(self.p, self.k, self.q)
        for i in range(m):
            if (i*x) % pk < m:
                j_ret.coef[(i*x) % pk] += self.coef[i]
            else:
                r = (i*x) % pk - self.p ** (self.k-1)                    
                while r>=0:
                    j_ret.coef[r] -= self.coef[i]
                    r-= self.p ** (self.k-1)
        return j_ret
    
                
    # operate sigma_x^(-1)
    def sigma_inv(self, x):
        m = self.m
        pk = self.pk
        j_ret=JacobiSum(self.p, self.k, self.q)
        for i in range(pk):
            if i<m:
                if (i*x)%pk < m:
                    j_ret.coef[i] += self.coef[(i*x)%pk]
            else:
                r = i - self.p ** (self.k-1)
                while r>=0:
                    if (i*x)%pk < m:
                        j_ret.coef[r] -= self.coef[(i*x)%pk]
                    r-= self.p ** (self.k-1)

        return j_ret
    

    # Is self p^k-th root of unity (mod N)
    # if so, return h where self is zeta^h
    def is_root_of_unity(self, N):
        m = self.m
        p = self.p
        k = self.k

        # case of zeta^h (h<m)
        one = 0
        for i in range(m):
            if self.coef[i]==1:
                one += 1
                h = i
            elif self.coef[i] == 0:
                continue
            elif (self.coef[i] - (-1)) %N != 0:
                return False, None
        if one == 1:
            return True, h

        # case of zeta^h (h>=m)
        for i in range(m):
            if self.coef[i]!=0:
                break
        r = i % (p**(k-1))
        for i in range(m):
            if i % (p**(k-1)) == r:
                if (self.coef[i] - (-1))%N != 0:
                    return False, None
            else:
                if self.coef[i] !=0:
                    return False, None

        return True, (p-1)*p**(k-1)+ r
         
# sum zeta^(a*x+b*f(x))
def calc_J_ab(p, k, q, a, b):
    j_ret = JacobiSum(p,k,q)
    g = None ; #  calculate f_q(x)  f = calc_f(q) find primitive root smallest_primitive_root(q)
    for r in range(2, q):
        s = set({})
        m = 1
        for i in range(1, q):
            m = (m*r) % q
            s.add(m)
        if len(s) == q-1:
            g = r
    m = {}
    for x in range(1,q-1):
        m[pow(g,x,q)] = x
    f = {}
    for x in range(1,q-1):
        f[x] = m[ (1-pow(g,x,q))%q ]
    for x in range(1,q-1):
        pk = p**k
        if (a*x+b*f[x]) % pk < j_ret.m:
            j_ret.coef[(a*x+b*f[x]) % pk] += 1
        else:
            r = (a*x+b*f[x]) % pk - p**(k-1)
            while r>=0:
                j_ret.coef[r] -= 1
                r-= p**(k-1)
    return j_ret

# Step 4
def APRtest_step4(p, k, q, N):
    if p>=3: #APRtest_step4a(p, k, q, N)
        J = calc_J_ab(p, k, q, 1, 1)
        # initialize s1=1
        s1 = JacobiSum(p,k,q).one()
        # J^Theta
        for x in range(p**k):
            if x % p == 0:
                continue
            t = J.sigma_inv(x)
            t = t.modpow(x, N)
            s1 = s1 * t
            s1.mod(N)
        # r = N mod p^k
        r = N % (p**k)
        # s2 = s1 ^ (N/p^k)
        s2 = s1.modpow(N//(p**k), N)
        # J^alpha
        J_alpha = JacobiSum(p,k,q).one()
        for x in range(p**k):
            if x % p == 0:
                continue
            t = J.sigma_inv(x)
            t = t.modpow((r*x)//(p**k), N)
            J_alpha = J_alpha * t
            J_alpha.mod(N)
        # S = s2 * J_alpha
        S = (s2 * J_alpha).mod(N)
        # Is S root of unity
        exist, h = S.is_root_of_unity(N)
        if not exist:
            # composite!
            result=False
            l_p=None
        else:
            # possible prime
            if h%p!=0:
                l_p = 1
            else:
                l_p = 0
            result=True
    elif p==2 and k>=3: # APRtest_step4b(p, k, q, N)
        j2q = calc_J_ab(p, k, q, 1, 1) # def calc_J3(p, k, q):calculate J_3(q)（p=2 and k>=3）
        j21 = calc_J_ab(p, k, q, 2, 1)
        J = j2q * j21 # J = calc_J3(p, k, q)
        # initialize s1=1
        s1 = JacobiSum(p,k,q).one()
        # J3^Theta
        for x in range(p**k):
            if x % 8 not in [1,3]:
                continue
            t = J.sigma_inv(x)
            t = t.modpow(x, N)
            s1 = s1 * t
            s1.mod(N)
        # r = N mod p^k
        r = N % (p**k)
        # s2 = s1 ^ (N/p^k)
        s2 = s1.modpow(N//(p**k), N)
        # J3^alpha
        J_alpha = JacobiSum(p,k,q).one()
        for x in range(p**k):
            if x % 8 not in [1,3]:
                continue
            t = J.sigma_inv(x)
            t = t.modpow((r*x)//(p**k), N)
            J_alpha = J_alpha * t
            J_alpha.mod(N)
        # S = s2 * J_alpha * J2^delta
        if N%8 in [1,3]:
            S = (s2 * J_alpha ).mod(N)
        else:
            j31 = calc_J_ab(2, 3, q, 3, 1) # def calc_J2(p, k, q): calculate J_2(q)（p=2 and k>=3）
            j_conv = JacobiSum(p, k, q)
            for i in range(j31.m):
                j_conv.coef[i*(p**k)//8] = j31.coef[i]
            J2_delta = j_conv * j_conv   
            S = (s2 * J_alpha * J2_delta).mod(N)
        # Is S root of unity
        exist, h = S.is_root_of_unity(N)
        if not exist:
            # composite 
            result=False
        else:
            # possible prime
            if h%p!=0 and (pow(q,(N-1)//2,N) + 1)%N==0:
                l_p = 1
            else:
                l_p = 0
            result=True
    elif p==2 and k==2: # APRtest_step4c(p, k, q, N)
        J2q = calc_J_ab(p, k, q, 1, 1)
        # s1 = J(2,q)^2 * q (mod N)
        s1 = (J2q * J2q * q).mod(N)
        # s2 = s1 ^ (N/4)
        s2 = s1.modpow(N//4, N)
        if N%4 == 1:
            S = s2
        elif N%4 == 3:
            S = (s2 * J2q * J2q).mod(N)
        else:
            print("Error")
        # Is S root of unity
        exist, h = S.is_root_of_unity(N)
        if not exist:
            # composite
            result=False
            l_p=None
        else:
            # possible prime
            if h%p!=0 and (pow(q,(N-1)//2,N) + 1)%N==0:
                l_p = 1
            else:
                l_p = 0
            result=True
    elif p==2 and k==1: #APRtest_step4d(p, k, q, N)
        S2q = pow(-q, (N-1)//2, N)
        if (S2q-1)%N != 0 and (S2q+1)%N != 0:
            # composite
            result=False
            l_p=None
        else:
            # possible prime
            if (S2q + 1)%N == 0 and (N-1)%4==0:
                l_p=1
            else:
                l_p=0
            result=True
    else:
        print("error")
    if not result:
     return False, None
    return result, l_p


def APRtest(N):

    if N<=3:
        return False
    # Select t
    for t in t_list:
        et, q_list = e(t)
        if N < et*et:
            break
    else:
        return False
    # Step 1
    g = gcd(t*et, N)
    if g > 1:
        return False
    # Step 2
    l = {}
    fac_t = prime_factorize(t)
    for p, k in fac_t:
        if p>=3 and pow(N, p-1, p*p)!=1:
            l[p] = 1
        else:
            l[p] = 0
    # Step 3 & Step 4
    for q in q_list:
        if q == 2:
            continue
        fac = prime_factorize(q-1)
        for p,k in fac:
            # Step 4
            result, l_p = APRtest_step4(p, k, q, N)
            if not result:
                # composite
                return False
            elif l_p==1:
                l[p] = 1
    # Step 5          
    for p, value in l.items():
        if value==0:
            # try other pair of (p,q)
            count = 0
            i = 1
            found = False
            # try maximum 30 times
            while count < 30:
                q = p*i+1
                if N%q != 0 and isprime_slow(q) and (q not in q_list):
                    count += 1
                    k = v(p, q-1)
                    # Step 4
                    result, l_p = APRtest_step4(p, k, q, N)
                    if not result:
                        # composite
                        return False
                    elif l_p == 1:
                        found = True
                        break
                i += 1
            if not found:
                return False
    # Step 6
    r = 1
    for t in range(t-1):
        r = (r*N) % et
        if r!=1 and r!= N and N % r == 0:
            return False
    # prime!!
    return True

########################################################################
def solve4thdeg():

 a=3
 b=1
 c=2
 d=4
 e=7

 t1=abs(2*c^3-9*b*c*d+27*a*d^2+27*b*e^2-72*a*c*e)
 t2=abs(t1+(-4*(c^2-3*b*d+12*a*e)^3+t1^2)**(1/2))
 t3=abs((c^2-3*b*d+12*a*e)/(3*a*(t2/2))**(1/3)+((t2/2)**(1/3)/(3*a)))
 t4=abs(((b^2)/(4*a^2)-(2*c/3*a)+t3)**(1/2))
 t5=abs((b^2)/(2*a^2)-(4*c/3*a)-t3)
 t6=abs(((-b^3)/(a**3)+(4*b*c)/(a^2)-(8*d)/a)/(4*t4) )
 x1=(-b/(4*a))+(t4/2)+((t5+t6)**(1/2))/2
 x2=(-b/(4*a))-(t4/2)+((t5-t6)**(1/2))/2
 x3=(-b/(4*a))+(t4/2)+((t5+t6)**(1/2))/2
 x4=(-b/(4*a))-(t4/2)+((t5-t6)**(1/2))/2



 print(" t1=",t1,sep='',end='')
 print(" t2=",t2,sep='',end='')
 print(" t3=",t3,sep='',end='')
 print(" t4=",t4,sep='',end='')
 print(" t5=",t5,sep='',end='')
 print(" t6=",t6,sep='',end='')

 print(" x1=",x1,sep='',end='')
 print(" x2=",x2,sep='',end='')
 print(" x3=",x3,sep='',end='')
 print(" x4=",x4,sep='',end='')

 print("x1=",x1,"=",nearToExact(abs(x1))) # or x1.real
 print("x2=",x2,"=",nearToExact(abs(x2)))
 print("x3=",x3,"=",nearToExact(abs(x3)))
 print("x4=",x4,"=",nearToExact(abs(x4)))

 return True
########################################################################

    
if __name__ == "__main__":

 
 while (True):

  print ("NumWorks SuperCharge        ")
  print (" a] order 2 Px k] unused-RFU")
  print (" b] order 3 Px l] unused-RFU")
  print (" c] order 4 Px m] unused-RFU") 
  print (" d] exact R->Q n] unused-RFU")
  print (" e] factorize  o] unused-RFU")
  print (" f] gcd scd    p] unused-RFU")
  print (" g] primality  q] unused-RFU")
  print (" h] pi digits  r] unused-RFU")
  print (" i] factorial  s] toggle led")
  print (" j] unused-RFU t] about     ")
  choice = input("Enter your choice [a-t] :  ")
 
  ### Convert string to int type ##
 # choice = int(choice)
   ### Take action as per selected menu-option ###
  if choice == 'd':
     prettyPrint(input("Enter a decimal number :"),None)
  elif choice == 'g':
   if APRtest(int(input("Enter a prime candidate :")))==True : 
    print("is prime")
   else:
    print("is composite")
  elif choice == 't':
         print ("version 3.4.1 Alex-Pauline Poudade (AlexPoudade@live.fr)")
  elif choice == 'c':
    solve4thdeg()
  else:    ## default ##
         print ("Invalid number. Try again...")




