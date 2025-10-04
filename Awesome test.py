import calculations as ca
import functions as fn

wp=1300 #what we give the ML

'''what it guesses'''
n1=2
Pnd=10
Np1=25
Helix=20

st1=ca.bending_stress(wp,n1,Pnd,Np1,Helix)
print(st1)
sat=36.8403

percent=fn.distance(st1,sat)
print(percent)

