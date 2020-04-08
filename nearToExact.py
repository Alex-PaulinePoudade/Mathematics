#convertit nombre approximé (réel flottant) en valeur exacte (rationnel)
#programmé par Alex-Pauline Poudade (AlexPoudade@live.fr) 08/03/2020

def pgcd (a,b):
    while b!=0:
        a,b=b,a%b
    return a

def num_after_comma(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

print ("Entrer un nombre fractionnaire : ")
n=input()

m=num_after_comma(n)
m=float(m)
n=float(n)
n2=10**m
n1=n*n2

if n1>n2:
    a=n1
    b=n2
    inverse=0
if n2>n1:
    a=n2
    b=n1
    inverse=1
if n1==n2:
   print ( n," est égal au rationnel : ",n1,"/","1")

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

print ( n," est égal au rationnel : ",int(Num),"/",int(Den))

    
