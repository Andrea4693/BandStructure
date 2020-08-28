# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:20:19 2020

@author: admin
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:11:27 2020

@author: andrea
"""


# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:27:36 2020

@author: Andrea Annunziata
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:42:17 2020

@author: Andrea Annunziata
"""
 

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cmath 
import scipy.constants as const
from scipy import fftpack
import time as tti
import skfuzzy.membership as fuzzy


t0=tti.time()
pi=np.pi
aSi=0.6101*10**(-9)
a=aSi/const.value('Bohr radius')                            #Lattice constant in a.u.
a=1                                                         #Lattice constant in a.u.
p1=a/8                                                      #dimensioni della buca
p2=7*a/8                                                    #Boundary
V0=-0.15                                                     #potenziale normalizzato
N=12                                                       #numero vettori base
p=p2-p1                                                     #ampiezza della buca
kpoints=100                                                #numero punti nello spazio k
#ESW=((const.hbar**2)*(const.pi**2))/(2*const.m_e*(a**2))    #Infinite square well energy
lamb=3200*10**(-9)                                          #Laser Wavelenght
w=2*np.pi*const.c/lamb                                      #Laser pulsation
freqau=const.value('atomic unit of velocity')/const.value('Bohr radius')
wau=w/freqau
timeau=const.value('Bohr radius')/const.value('atomic unit of velocity')
T=16*np.pi/w
Td=16*np.pi/wau                                               #Pulse Duration in a.u.
Iau=3.51*10**16                                             #Atomic unit laser intensity
I0=1*10**13                                                 #Laser peak intensity
E0=np.sqrt(I0/Iau)                                          #Electric Field peak intensity
A0=E0/wau                                                      #Assuming I0=Iau -> E0=1
#A0=0.304#inserito per provare paper
#%%#Create p and q used for quant number creare le coppie (0,0) (0,1) (0,-1) etc
j=np.arange(1,N+2)                                          #array per creare la alpha 

nn=(1+np.power(-1,j)*(2*j-1))/4                             #array contenenti i numeri d'onda
pp, qq=np.meshgrid(nn,nn)
i=0
n=np.zeros([(N+1)**2,2])
for i in range(0,N+1):
    n[i*(N+1):(N+1)+i*(N+1),0]=pp[:,i]
    n[i*(N+1):(N+1)+i*(N+1),1]=qq[:,i]
nx=n[:,0]                                                   #Crea nx ed ny 
ny=n[:,1]
n2=np.zeros(np.size(nx))                                    #Crea numeri quantici per energia
nn=np.zeros([np.size(nx),3])                                #in ordine crescente
nn[:,1]=n[:,0]                               
nn[:,2]=n[:,1]
nn[:,0]=np.square(n[:,0])+np.square(n[:,1])                 #mette nella prima colonna le energie
                                                            #somma di nx ed ny al quadrato 
                                                            #associato ad ogni coppia di numeri 
                                                            #quantici
                                            
#riordino la tabella degli indici in modo da mettere le energie crescenti e 
#con affianco i numeri quantici che generano tale energia
sortindex=np.argsort(nn[:,0])                               #fornisce indici dell'ordinamento
nsorted=np.zeros(np.shape(nn))                              #alloca spazio ad ordinata
nsorted[:,0]=np.sort(nn[:,0])                               #prima col. sono le energie
j=0
for i in sortindex:
    ntemp=nn[i,1:3]                                         #estrae riga i 
    nsorted[j,1:3]=ntemp                                    #la mette uguale a indice sorting
    j+=1 

#ridefinisco nx ed ny con ordine crescente
nx=nsorted[:,1]
ny=nsorted[:,2]
n2=nsorted[:,0]
nx=np.around(nx)
ny=np.around(ny)
n2=np.around(n2)
nx=nx.astype(int)
ny=ny.astype(int)
n2=n2.astype(int)

del  n, pp, qq, j, i, ntemp, sortindex, nn
#%% Costruisco Hamiltoniana

def ofdi(mx,nx):                                            #function for offdiagonal elements
    if mx != nx:
        return (1/(1j*2*np.pi*(mx-nx)))*(np.exp(1j*(2*np.pi)*p2*(mx-nx))-
                                             np.exp(1j*(2*np.pi)*p1*(mx-nx)))
    else:
        return p                                            #when index equal, V contribution 
    
mx=nx                                                       #define new set of index for calc
my=ny                                                       #the differences

h0=np.diag((np.pi**2/(a**2)) *2* n2+V0*p**2)                                    #elementi sulla diag principale
Ix=np.zeros(1,dtype=complex)
h1=np.zeros(np.shape(h0),dtype=complex)
h11=np.zeros(np.shape(h0))
for i in range(0,(N+1)**2):
    for j in range(i+1,(N+1)**2):
        Ix=ofdi(mx[j],nx[i])
        Iy=ofdi(my[j],ny[i])
        h1[i,j]=Ix*Iy*V0
        
        
h0=h1+h0#farlo direttamente nel ciclo eliminando h1
#For Simmetry of H, lower triangular equal to the upper one (dagger)
h0=h0+np.conjugate(np.transpose(np.triu(h0,1)))                           #Field free Hamiltonian

#%%Generazione spazio K
Kx=np.arange(0,np.pi+np.pi/kpoints,np.pi/kpoints)
Ky=np.arange(0,np.pi+np.pi/kpoints,np.pi/kpoints)
m=np.arange(0,np.size(nx))
nnx=nx[m]
nny=ny[m]

#Elementi da gamma ad X' (vanno da 0,0 a 0,pi/ay, nota ora ay=ax=1) nota elementi nx^2 ed ny^2 sono
#già calcolati nei termini diagonali della matrice nella sezione precedente

psibloch=np.zeros(np.size(nx))
GX=np.zeros([kpoints+1,np.size(nx)])
for i in range(0,kpoints+1):
     psibloch=np.pi**2/(2*a**2) * ( (Kx[0]**2)*a**2/pi**2 + 4*nnx*Kx[0]*a/pi + 
                                    (Ky[i]**2)*a**2/pi**2 + 4*nny*Ky[i]*a/pi )
     eig=la.eig(h0+np.diag(psibloch))                         #calcola autovalori
     eigsort=np.sort(eig[0])                                 #ascending order
     GX[i,:]=eigsort                                         #each row different ky
     
#Elementi da X' ad M (va da 0,pi/ay fino a pi/ax pi/ay)
psibloch=np.zeros(np.size(nx))
XM=np.zeros(np.shape(GX))                                   #from X' to M 
for i in range(0,kpoints+1):
      psibloch=np.pi**2/(2*a**2) * ( (Kx[i]**2)*a**2/pi**2 + 4*nnx*Kx[i]*a/pi + 
                                    (Ky[-1]**2)*a**2/pi**2 + 4*nny*Ky[-1]*a/pi )
      eig=la.eig(h0+np.diag(psibloch))                        #calcola autovalori
      eigsort=np.sort(eig[0])                                #sor in ascending order
      XM[i,:]=eigsort                                        #each row different kx

#Elementi che vanno da M a Gamma (ritorno in 00), mette i valori sulla diag
#della matrice con i valori di E per ogni kx e ky
psibloch=np.zeros(np.size(nx))
MG=np.zeros(np.shape(GX))                                    #from X' to M 
for i in range(kpoints,-1,-1):                               #si ferma a 0 non a -1
    psibloch=np.pi**2/(2*a**2) * ( (Kx[i]**2)*a**2/pi**2 + 4*nnx*Kx[i]*a/pi + 
                                    (Ky[i]**2)*a**2/pi**2 + 4*nny*Ky[i]*a/pi )
    eig=la.eig(h0+ np.diag(psibloch))                       #Find the eigenvalues
    eigsort=np.sort(eig[0])                                 #Sort in ascending order
    MG[i,:]=eigsort                                         #Each row different kx ky



#%% Extraction of element for plotting

#NOTA, K along the main diagonal is scaled by a factor of sqrt(2)
#-0.66.. makes it as starting from initial point,
# then it takes into account the sqrt(2) factor for the spacing
K= np.arange(0,1+1/(3*kpoints),1/(3*kpoints))
for i in range(1+2*kpoints, np.size(K)):                    #elements where add sqrt(2)
    K[i]=K[i] + np.sqrt(2)*0.3333*(K[i] - 0.66667)

#array for the first quantum numbers
E0=np.concatenate((GX[:,0],XM[1:-1,0],MG[:,0][::-1]))
E1=np.concatenate((GX[:,1],XM[1:-1,1],MG[:,1][::-1]))
E2=np.concatenate((GX[:,2],XM[1:-1,2],MG[:,2][::-1]))
E3=np.concatenate((GX[:,3],XM[1:-1,3],MG[:,3][::-1]))
E4=np.concatenate((GX[:,4],XM[1:-1,4],MG[:,4][::-1]))
E5=np.concatenate((GX[:,5],XM[1:-1,5],MG[:,5][::-1]))
E6=np.concatenate((GX[:,6],XM[1:-1,6],MG[:,6][::-1]))
E7=np.concatenate((GX[:,7],XM[1:-1,7],MG[:,7][::-1]))
E8=np.concatenate((GX[:,8],XM[1:-1,8],MG[:,8][::-1]))
E9=np.concatenate((GX[:,9],XM[1:-1,9],MG[:,9][::-1]))
plt.figure()
plt.plot(K,E0)
plt.plot(K,E1)
plt.plot(K,E2)
plt.plot(K,E3)
plt.plot(K,E4)
plt.plot(K,E5)
plt.plot(K,E6)
plt.plot(K,E7)
plt.plot(K,E8)
plt.plot(K,E9)

#%% Create the field decribed through the vector potential, Td is in a.u.
#supponendo di voler osservare fino alla 20a armonica
wmax=20*wau                         #Maximum frequency
Fn=2*wmax                           #Nyquist frequency
t=np.arange(0,Td,1/Fn)
dt=t[1]-t[0]
theta=pi/4
t2=t+dt
fs=(Fn/np.size(t))
f=np.arange(0,wmax,fs)
fev=const.value('reduced Planck constant in eV s')*freqau*f
Nsample=np.size(t)
#restano da definire fuzzy e gbell
def field(A0,Td,w,t):                                       #potenziale vettore assumendo 
    return A0 *(((np.cos((t-Td/2)*pi/Td))**2)* np.sin(w*(t-Td/2)) * fuzzy.gaussmf(t-Td/2,0,Td/4) *
                                                 fuzzy.gbellmf(t-Td/2,Td/4,7,0))        #inviluppo cos^2

At=field(A0,Td,wau,t2)

def fieldxy(theta):
    return At*np.cos(theta), At*np.sin(theta)
Ax, Ay = fieldxy(theta)
plt.figure()
plt.title("Ax")
plt.plot(t2,Ax)
plt.figure()
plt.title("Ay")
plt.plot(t2,Ay)
At_spectrumx=np.fft.fftshift(np.fft.fft(Ax))
At_spectrumy=np.fft.fftshift(np.fft.fft(Ay))

plt.figure()
plt.title("Ax Spectrum")
plt.plot(fev[0:15],At_spectrumx[int(Nsample/2):int(Nsample/2)+15])
plt.figure()
plt.title("Ay Spectrum")
plt.plot(fev[0:15],At_spectrumy[int(Nsample/2):int(Nsample/2)+15])

#%% DOVREBBE ESSERE COSI
vec=np.zeros((N+1)**2,dtype=complex)   
vec=eig[1] #contiene tutti gli autovettori
eigindex=np.argsort(eig[0])     #cerca quello della banda di valenza
eigvec_valence=vec[:,eigindex[0]]
eigvec_valence_plus1=vec[:,eigindex[1]]
N = 101
N_middle = 51
x=np.linspace(0,a,N)
X, Y =np.meshgrid(x,x)

blochwave=np.zeros(np.shape(X),dtype=complex)
blochwave_valence = np.zeros(np.shape(X))
blochwave_valence_plus1 = np.zeros(np.shape(X))
for ii, (nx_ii, ny_ii) in enumerate(zip(nnx, nny)): #ii indice generico di enumerate, nx ed ny elementi dati da zip
    planwave_ii = np.exp(1j*(2*pi/a)*ny_ii*Y)*np.exp(1j*(2*pi/a)*nx_ii*X) #per ogni xy crea la funzione d'onda per una data coppia di nx ed ny
    blochwave = blochwave + eigvec_valence[ii]*planwave_ii/a
    blochwave_valence_plus1 = blochwave_valence_plus1 + eigvec_valence_plus1[ii]*planwave_ii/a

#%% Plot della funzione d'onda
plt.figure()
plt.imshow(np.abs(blochwave))
plt.figure()
plt.plot(x, np.abs(blochwave[N_middle,:]))
plt.plot(x, np.abs(blochwave_valence_plus1[N_middle,:]))
plt.plot(x, np.angle(blochwave[N_middle,:])) #np.angle da la fase
plt.plot(x, np.angle(blochwave_valence_plus1[N_middle,:]))
plt.legend(["absolute value valence","absolute value conduction", "phase valence", "phase conduction"])
#%% Crank-Nicholson

M=np.zeros(np.shape(h0),dtype=complex) #+1 ->prima colonna autovettori originali
C=np.zeros([np.size(h0,0),np.size(t)],dtype=complex)

C0=eigvec_valence

def Hi(Axx,Ayy):
    return np.diag( -( (2*pi/a)*nnx*Axx + (2*pi/a)*nny*Ayy ) )
C[:,0]=C0
for i in range(0,len(t)-1):
    M=np.dot(la.inv((np.eye(np.size(h0,0)) + (1j*(h0+Hi(Ax[i],Ay[i])))*dt/2)) ,
             (np.eye(np.size(h0,0)) - ((1j*(h0+Hi(Ax[i],Ay[i])))*dt/2)))
    C[:,i+1]=np.dot(M,C[:,i])
    

jx=np.zeros(len(t),dtype=complex)
jy=np.zeros(len(t),dtype=complex)
norm=np.zeros(len(t),dtype=complex)

for i in range(len(t)): 
    for c,nxx,nyy in zip(C[:,i],nnx,nny):
        jx[i]+=-(np.conjugate(c)*c)*nxx  * (2*pi/a)
        jy[i]+=-(np.conjugate(c)*c)*nyy  * (2*pi/a)
        norm[i]+=np.conj(c)*c                       #check normalization

jfx=np.fft.fftshift( np.fft.fft((( jx.real * np.hanning(np.size(t))))))
jfy=np.fft.fftshift(np.fft.fft((( jy.real * np.hanning(np.size(t))))))
j=(np.sqrt(np.abs(jfx)**2+np.abs(jfy)**2))
HHG=np.conjugate(j)*j
HHGx=np.conjugate(jfx)*jfx
HHGy=np.conjugate(jfy)*jfy
plt.figure()
plt.title("Current along x vs time")
plt.plot(t,jx) 

plt.figure()
plt.title("Current along y vs time")
plt.plot(t,jy)       

plt.figure()
plt.title("HHG Spectrum (lambda=%i nm)" %(lamb*10**9))
plt.yscale('log')
plt.plot(fev[0:-1],HHG[int(Nsample/2):-1].real)
plt.xticks(np.arange(min(fev),max(fev),1))
plt.xlabel('Photons energy [ev]')
plt.grid()

plt.figure()
plt.title("HHG Spectrum Section (lambda=%i nm) " %(lamb*10**9))
plt.yscale('log')
plt.plot(fev[1:200],HHG[int(Nsample/2)+1:int(Nsample/2)+200].real) #Plot per fare un po' di zoom
plt.xticks(np.arange(min(fev),max(fev[0:200]),0.1)) #per vedere che escano solo armoniche dispari
plt.xlabel('Photons energy [ev]')
plt.grid()




t1=tti.time()
elapsed_time=t1-t0

