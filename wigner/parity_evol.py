#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:20:11 2019

@author: clarkesmith
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import qutip as qt
import seaborn as sns

from scipy import special
from scipy import linalg

sns.set()
cmap = sns.diverging_palette(240, 10, s=99, l=30, as_cmap=True)

'''
Physical constants
'''

h = 6.626e-34
hbar = h/(2*np.pi)
e = 1.602e-19
phi0 = h/(2*e)
phi0bar = phi0/(2*np.pi)
gap = 0.00017*e #from Kittel

'''
System parameters
'''

# H = hbar*omega*(adagger*a - a*cos(x*(a + adagger)))
a = 0.01
x = 4

params = [a,x]
dim = 30

'''
Functions for converting energies
'''

def cap(EC):
    EC = EC*1e9*h
    return e**2/(2*EC)

def ind(EL):
    EL = EL*1e9*h
    return phi0bar**2/EL

def enc(C):
    C = C*1e9*h
    return e**2/(2*C)
    
def enl(L):
    L = L*1e9*h
    return phi0bar**2/L

def phizpf(L, C):
    EL = enl(L)
    EC = enc(C)
    return (2*EC/EL)**0.25

def Nzpf(L, C):
    EL = enl(L)
    EC = enc(C)
    return 0.5*(0.5*EL/EC)**0.25

'''
Generic functions
'''

def matel(x, i, j):
    n = min(i, j)
    m = max(i, j)
    factor = (-0.5)**((m-n)/2.) \
            *1./np.sqrt(special.poch(n+1,m-n)) \
            *x**(m-n) \
            *np.exp(-0.25*x**2) \
            *special.eval_genlaguerre(n, m-n, 0.5*x**2)
    return factor

def cosx(p, i, j):
    if (i-j)%2 == 0:
        return matel(p, i, j)
    else:
        return 0
    
def sinx(p, i, j):
    if (i-j)%2 == 1:
        comp = -1j*matel(p, i, j)
        if abs(np.imag(comp)) > 1e-10:
            raise ValueError("Matrix element is complex")
        else:
            return np.real(comp)
    else:
        return 0
    
def cosp(x, i, j):
    if (i-j)%2 == 0:
        return (-1.)**((i-j)/2.)*matel(x, i, j)
    else:
        return 0
    
def sinp(x, i, j):
    if (i-j)%2 == 1:
        comp = (-1.)**((i-j)/2.)*matel(x, i, j)
        if abs(np.imag(comp)) > 1e-10:
            raise ValueError("Matrix element is complex")
        else:
            return -1j*np.real(comp)
    else:
        return 0

# convention: sin(1j*alpha*adagger - 1j*alpha.conj*a)
# convention: cos(1j*alpha*adagger - 1j*alpha.conj*a)
def mats(dim, alpha):
    sin_im = np.zeros((dim,dim))
    cos_im = np.zeros((dim,dim))
    sin_re = np.zeros((dim,dim), dtype=np.cfloat)
    cos_re = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            cos_im[i,j] += cosx(np.sqrt(2)*np.imag(alpha),i,j)
            cos_re[i,j] += cosp(np.sqrt(2)*np.real(alpha),i,j)
            sin_im[i,j] += sinx(np.sqrt(2)*np.imag(alpha),i,j)
            sin_re[i,j] += sinp(np.sqrt(2)*np.real(alpha),i,j)
    return sin_im, cos_im, sin_re, cos_re

def displacement(dim, alpha):
    sin_im, cos_im, sin_re, cos_re = mats(dim, alpha)
    BCH = np.exp(-1j*np.real(alpha)*np.imag(alpha))
    im = cos_im + 1j*sin_im
    re = cos_re - 1j*sin_re
    return  BCH*np.dot(im, re)

def coherent(dim, alpha):
    vec = np.zeros(dim,dtype=np.cfloat)
    for i in range(dim):
        vec[i] += alpha**i/np.sqrt(special.factorial(i))
    vec *= np.exp(-0.5*abs(alpha)**2)
    return vec

def fock(dim, idx):
    vec = np.zeros(dim)
    vec[idx] = 1.0
    return vec

def parity(dim):
    par = np.zeros((dim,dim),dtype=np.cfloat)
    for i in range(dim):
        par[i,i] = np.exp(1j*np.pi*i)
    return par

def disp_parity(dim, alpha):
    par = parity(dim)
    disp = displacement(dim, alpha)
    return np.dot(disp, np.dot(par, disp.conj().T))

def wigner(vec, I, Q, verbose=True):
    dim = len(vec)
    W = np.zeros((len(I),len(Q)), dtype=np.cfloat)
    for i, x in enumerate(I):
        for j, y in enumerate(Q):
            if verbose == True:
                if (i*len(Q_mesh) + j) % int(len(I_mesh)*len(Q_mesh)/10) == 0:
                    print('.',end='')
            beta = x + 1j*y
            dp = disp_parity(dim, beta)
            W[i,j] += np.vdot(state, np.dot(dp, vec))
    return W

'''
Specific functions
'''

def ham(dim, params, diag=True, offdiag=True):
    a, x = params
    H = np.zeros((dim,dim))
    for i in range(dim):
        H[i,i] += i
        for j in range(dim):
            if (i-j)%2 == 0:
                if diag and i == j:
                    H[i,j] -= a*cosx(np.sqrt(2)*x,i,j)
                elif offdiag and i != j:
                    H[i,j] -= a*cosx(np.sqrt(2)*x,i,j)
    return H

def ham_kerr(dim, params, coincide=False):
    a, x = params
    H = np.zeros((dim,dim))
    for i in range(dim):
        if coincide:
            H[i,i] += (1.0 + a*x**2 - 0.25*a*x**4)*i
            H[i,i] -= 0.25*a*x**4*i**2
        else:
            H[i,i] += (1.0 + a)*i
            H[i,i] -= a*i**2
    return H

def ham_parity(dim, params):
    a, x = params
    H = np.zeros((dim,dim))
    for i in range(dim):
        H[i,i] += i
        H[i,i] -= a*np.exp(-0.5*x**2)*special.eval_laguerre(int(0.25*x**2),x**2)*np.cos(np.pi*i)
    return H

def Uharm(t, dim):
    U = np.zeros((dim,dim),dtype=np.cfloat)
    for i in range(dim):
        U[i,i] += np.exp(-1j*i*t)
    return U

def Utrue(t, dim, params, **kwargs):
    H = ham(dim, params, **kwargs)
    return linalg.expm(-1j*H*t)

def U_kerr(t, dim, params, **kwargs):
    H = ham_kerr(dim, params, **kwargs)
    return linalg.expm(-1j*H*t)

def U_parity(t, dim, params):
    H = ham_parity(dim, params)
    return linalg.expm(-1j*H*t)
    

'''
Calculations
'''

if 0: # qutip diagonalization
    H = qt.num(dim) - 0.5*a*(qt.displace(dim,1j*x) + qt.displace(dim,-1j*x))
    ev = H.eigenenergies()
    
    if 1: # plot
        fig, ax = plt.subplots()
        ax.plot(range(dim-1),ev[1:]-ev[:-1],'o')
        ax.plot(range(dim-1),ev[1:]-ev[:-1])
        ax.set_xlim(0,4*x)

if 0: # manual diagonalization
    H = ham(dim, params)
    ev, ek = linalg.eigh(H)
    
    if 1: # plot
        fig, ax = plt.subplots()
        ax.plot(range(dim-1),ev[1:]-ev[:-1],'o')
        ax.plot(range(dim-1),ev[1:]-ev[:-1])
        ax.set_xlim(0,dim)
    
if 0: # test displacement operator
    alpha = 2 + 3j
    state = coherent(dim, alpha)
    
    vac = coherent(dim, 0)
    disp = displacement(dim, alpha)
    displace = np.dot(disp, vac)
    
    fig, ax = plt.subplots()
    ax.plot(range(dim),abs(displace))
    ax.plot(range(dim),abs(state),'o')
    
if 0: # qutip wigner of a coherent state
    time0 = time.time()
    
    alpha = 2*np.exp(1j*np.pi/3)
    state = qt.coherent(dim, alpha)
    
    I_mesh = np.linspace(-5,5,51)
    Q_mesh = np.linspace(-5,5,101)
    
    W = qt.wigner(state, I_mesh, Q_mesh, g=2)
    
    time1 = time.time()
    print("qutip time = ",time1-time0)
    
    if 1: # plot
        fig, ax = plt.subplots()
        cs = ax.imshow(W,origin='lower',extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)])
        ax.plot(np.real(alpha),np.imag(alpha),'r+')
        plt.grid()
        plt.colorbar(cs)
    
if 0: # manual wigner of a coherent state
    time0 = time.time()
    
    alpha = 2*np.exp(1j*np.pi/3)
    state = coherent(dim, alpha)
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    W = wigner(state, I_mesh, Q_mesh)
                
    time1 = time.time()
    print("manual time = ",time1-time0)
    
    if 1: # plot
        fig, ax = plt.subplots()
        cs = ax.imshow(np.real(W).T,origin='lower',extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)])
        ax.plot(np.real(alpha),np.imag(alpha),'r+')
        plt.grid()
        plt.colorbar(cs)
    
if 0: # wigner of a coherent state under Hamiltonian evolution
    alpha = 2*np.exp(1j*np.pi/3)
    Istate = coherent(dim, alpha)
    
    t = np.linspace(0.0,100.0,401) # units of 1/omega
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    ims = []
    fig = plt.figure()
    
    for i, T in enumerate(t):
        if i % int(len(t)/10) == 0:
            print('.',end='')
        
        U = Utrue(T, dim, params)
        state = np.dot(U,Istate)
        state = qt.Qobj(state)
    
        W = qt.wigner(state, I_mesh, Q_mesh, g=2)
        
        if T == t[0]:
            scale = np.amax(abs(W))
    
        im = plt.imshow(W, origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                   animated=True, vmin=-scale, vmax=+scale, cmap=cmap)
        lab = plt.text(max(I_mesh),min(Q_mesh),"%.2f" % T + r"$/\omega$")
        ims.append([im,lab])
    
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.plot(np.real(alpha),np.imag(alpha),'r+')
    plt.colorbar(im, shrink=0.8)
    plt.show()
    
if 0: # wigner of a coherent state under Hamiltonian evolution in rotating frame
    alpha = 2*np.exp(1j*np.pi/3)
    Istate = coherent(dim, alpha)
    
    t = np.linspace(0.0,10000.0,401) # units of 1/omega
    #t = np.linspace(0.0,10000.0,401)
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    ims = []
    fig = plt.figure()
    
    for i, T in enumerate(t):
        if i % int(len(t)/10) == 0:
            print('.',end='')
        
        U = Utrue(T, dim, params, offdiag=False)
        Urotate = Uharm(T, dim).conj().T
        Ueff = np.dot(Urotate,U)
        state = np.dot(Ueff,Istate)
        state = qt.Qobj(state)
    
        W = qt.wigner(state, I_mesh, Q_mesh, g=2)
        
        if T == t[0]:
            scale = np.amax(abs(W))
    
        im = plt.imshow(W, origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                   animated=True, vmin=-scale, vmax=+scale, cmap=cmap)
        lab = plt.text(max(I_mesh),min(Q_mesh),"%.2f" % T + r"$/\omega$")
        ims.append([im,lab])
    
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.plot(np.real(alpha),np.imag(alpha),'r+')
    plt.colorbar(im, shrink=0.8)
    plt.show()
    
if 0: # wigner of a coherent state under different evolutions after some time
    alpha = 2*np.exp(1j*np.pi*0)
    Istate = coherent(dim, alpha)
    
    T = 300.0 # units of 1/omega
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    params_1 = [a,3.5]
    params_2 = [a,4.5]
    
    U = [Utrue(T, dim, params),
         Utrue(T, dim, params_1),
         Utrue(T, dim, params_2),
         U_parity(T, dim, params)]
    Urotate = Uharm(T, dim).conj().T
    
    W = []
    
    for u in U:
        Ueff = np.dot(Urotate,u)
        state = np.dot(Ueff,Istate)
        state = qt.Qobj(state)
        
        W.append(qt.wigner(state, I_mesh, Q_mesh, g=2))
    
    scale = np.amax(abs(W[0]))

    fig, ax = plt.subplots(2,2)
    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].imshow(W[2*i+j], origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                             vmin=-scale, vmax=+scale, cmap=cmap)
    
if 0: # wigner of a cat state under Hamiltonian evolution in rotating frame
    alpha = 2*np.exp(1j*np.pi*0)
    Istate = (coherent(dim, alpha) + coherent(dim, -alpha))/np.sqrt(2)
    
    t = np.linspace(0.0,10000.0,401) # units of 1/omega
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    ims = []
    fig = plt.figure()
    
    for i, T in enumerate(t):
        if i % int(len(t)/10) == 0:
            print('.',end='')
        
        #U = Uharm(T, dim)
        #U = Utrue(T, dim, params)
        U = Utrue(T, dim, params, diag=True)
        Urotate = Uharm(T, dim).conj().T
        Ueff = np.dot(Urotate,U)
        state = np.dot(Ueff,Istate)
        state = qt.Qobj(state)
    
        W = qt.wigner(state, I_mesh, Q_mesh, g=2)
        
        if T == t[0]:
            scale = np.amax(abs(W))
    
        im = plt.imshow(W, origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                   animated=True, vmin=-scale, vmax=+scale, cmap=cmap)
        lab = plt.text(max(I_mesh),min(Q_mesh),"%.2f" % T + r"$/\omega$")
        ims.append([im,lab])
    
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.plot(np.real(alpha),np.imag(alpha),'r+')
    plt.plot(-np.real(alpha),-np.imag(alpha),'r+')
    plt.colorbar(im, shrink=0.8)
    plt.show()
    
if 0: # wigner of a cat state under different evolutions after some time
    alpha = 2*np.exp(1j*np.pi*0)
    Istate = (coherent(dim, alpha) + coherent(dim, -alpha))/np.sqrt(2)
    
    T = 200.0 # units of 1/omega
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    params_1 = [a,3.5]
    params_2 = [a,4.5]
    
    U = [Utrue(T, dim, params),
         Utrue(T, dim, params_1),
         Utrue(T, dim, params_2),
         U_parity(T, dim, params)]
    Urotate = Uharm(T, dim).conj().T
    
    W = []
    
    for u in U:
        Ueff = np.dot(Urotate,u)
        state = np.dot(Ueff,Istate)
        state = qt.Qobj(state)
        
        W.append(qt.wigner(state, I_mesh, Q_mesh, g=2))
    
    scale = np.amax(abs(W[0]))

    fig, ax = plt.subplots(2,2)
    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].imshow(W[2*i+j], origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                             vmin=-scale, vmax=+scale, cmap=cmap)
    
if 0: # wigner of a fock state under Hamiltonian evolution in rotating frame
    idx = 15
    Istate = fock(dim, idx)
    
    t = np.linspace(0.0,10000.0,51) # units of 1/omega
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    ims = []
    fig = plt.figure()
    
    for i, T in enumerate(t):
        if i % int(len(t)/10) == 0:
            print('.',end='')
        
        #U = Uharm(T, dim)
        U = Utrue(T, dim, params)
        Urotate = Uharm(T, dim).conj().T
        Ueff = np.dot(Urotate,U)
        state = np.dot(Ueff,Istate)
        
        state = qt.Qobj(state)
    
        W = qt.wigner(state, I_mesh, Q_mesh, g=2)
        
        if T == t[0]:
            scale = np.amax(abs(W))
    
        im = plt.imshow(W, origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                   animated=True, vmin=-scale, vmax=+scale, cmap=cmap)
        lab = plt.text(max(I_mesh),min(Q_mesh),"%.2f" % T + r"$/\omega$")
        ims.append([im,lab])
    
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.colorbar(im, shrink=0.8)
    plt.show()
    
if 0: # matrix elements of perturbation
    shift = 4 # distance from diagonal
    x_mesh = np.linspace(0.0,10,101)
    n_mesh = range(10)
    
    mat_diag = np.zeros((len(n_mesh),len(x_mesh)))
    for i in n_mesh:
        mat_diag[i] = -a*cosx(np.sqrt(2)*x_mesh,i,i+shift)
    
    fig, ax = plt.subplots()
    for i in n_mesh:
        ax.plot(x_mesh, mat_diag[i])
        
if 0: # wigner of a coherent state under Kerr evolution in rotating frame
    alpha = 2*np.exp(1j*np.pi/3)
    Istate = coherent(dim, alpha)
    
    t = np.linspace(0.0,1000.0,401) # units of 1/omega
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    ims = []
    fig = plt.figure()
    
    for i, T in enumerate(t):
        if i % int(len(t)/10) == 0:
            print('.',end='')
        
        U = U_kerr(T, dim, params, coincide=False)
        Urotate = Uharm(T, dim).conj().T
        Ueff = np.dot(Urotate,U)
        state = np.dot(Ueff,Istate)
        state = qt.Qobj(state)
    
        W = qt.wigner(state, I_mesh, Q_mesh, g=2)
        
        if T == t[0]:
            scale = np.amax(abs(W))
    
        im = plt.imshow(W, origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                   animated=True, vmin=-scale, vmax=+scale, cmap=cmap)
        lab = plt.text(max(I_mesh),min(Q_mesh),"%.2f" % T + r"$/\omega$")
        ims.append([im,lab])
    
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.plot(np.real(alpha),np.imag(alpha),'r+')
    plt.colorbar(im, shrink=0.8)
    plt.show()
    
if 0: # wigner of a cat state under Kerr evolution in rotating frame
    alpha = 2*np.exp(1j*np.pi*0)
    Istate = (coherent(dim, alpha) + coherent(dim, -alpha))/np.sqrt(2)
    
    t = np.linspace(0.0,1000.0,401) # units of 1/omega
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    ims = []
    fig = plt.figure()
    
    for i, T in enumerate(t):
        if i % int(len(t)/10) == 0:
            print('.',end='')
        
        U = U_kerr(T, dim, params)
        Urotate = Uharm(T, dim).conj().T
        Ueff = np.dot(Urotate,U)
        state = np.dot(Ueff,Istate)
        state = qt.Qobj(state)
    
        W = qt.wigner(state, I_mesh, Q_mesh, g=2)
        
        if T == t[0]:
            scale = np.amax(abs(W))
    
        im = plt.imshow(W, origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                   animated=True, vmin=-scale, vmax=+scale, cmap=cmap)
        lab = plt.text(max(I_mesh),min(Q_mesh),"%.2f" % T + r"$/\omega$")
        ims.append([im,lab])
    
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.plot(np.real(alpha),np.imag(alpha),'r+')
    plt.plot(-np.real(alpha),-np.imag(alpha),'r+')
    plt.colorbar(im, shrink=0.8)
    plt.show()
        
if 0: # wigner of a coherent state under parity evolution in rotating frame
    alpha = 2*np.exp(1j*np.pi/3)
    Istate = coherent(dim, alpha)
    
    t = np.linspace(0.0,1000.0,101) # units of 1/omega
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    ims = []
    fig = plt.figure()
    
    for i, T in enumerate(t):
        if i % int(len(t)/10) == 0:
            print('.',end='')
        
        U = U_parity(T, dim, params)
        Urotate = Uharm(T, dim).conj().T
        Ueff = np.dot(Urotate,U)
        state = np.dot(Ueff,Istate)
        state = qt.Qobj(state)
    
        W = qt.wigner(state, I_mesh, Q_mesh, g=2)
        
        if T == t[0]:
            scale = np.amax(abs(W))
    
        im = plt.imshow(W, origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                   animated=True, vmin=-scale, vmax=+scale, cmap=cmap)
        lab = plt.text(max(I_mesh),min(Q_mesh),"%.2f" % T + r"$/\omega$")
        ims.append([im,lab])
    
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.plot(np.real(alpha),np.imag(alpha),'r+')
    plt.colorbar(im, shrink=0.8)
    plt.show()
    
if 0: # wigner of a cat state under parity evolution in rotating frame
    alpha = 2*np.exp(1j*np.pi*0)
    Istate = (coherent(dim, alpha) + coherent(dim, -alpha))/np.sqrt(2)
    
    t = np.linspace(0.0,1000.0,401) # units of 1/omega
    
    I_mesh = np.linspace(-5,5,101)
    Q_mesh = np.linspace(-5,5,101)
    
    ims = []
    fig = plt.figure()
    
    for i, T in enumerate(t):
        if i % int(len(t)/10) == 0:
            print('.',end='')
        
        U = U_parity(T, dim, params)
        Urotate = Uharm(T, dim).conj().T
        Ueff = np.dot(Urotate,U)
        state = np.dot(Ueff,Istate)
        state = qt.Qobj(state)
    
        W = qt.wigner(state, I_mesh, Q_mesh, g=2)
        
        if T == t[0]:
            scale = np.amax(abs(W))
    
        im = plt.imshow(W, origin='lower', extent=[min(I_mesh),max(I_mesh),min(Q_mesh),max(Q_mesh)], \
                                                   animated=True, vmin=-scale, vmax=+scale, cmap=cmap)
        lab = plt.text(max(I_mesh),min(Q_mesh),"%.2f" % T + r"$/\omega$")
        ims.append([im,lab])
    
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.plot(np.real(alpha),np.imag(alpha),'r+')
    plt.plot(-np.real(alpha),-np.imag(alpha),'r+')
    plt.colorbar(im, shrink=0.8)
    plt.show() 