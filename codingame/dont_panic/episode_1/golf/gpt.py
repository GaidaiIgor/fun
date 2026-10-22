C=int;S=lambda:input().split()
*_,e,x,_,_,n=map(C,S())
m=dict(map(C,S())for _ in[0]*n)|{e:x}
while 1:f,p,d=S();f=C(f);p=C(p);print(("WAIT","BLOCK")[~f and(p-m[f])*(d>"M"or-1)>0])
