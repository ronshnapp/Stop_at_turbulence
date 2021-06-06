# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 10:33:33 2021

@author: Ron
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt



a = np.random.normal(0,1,(5000,5000))

for i,exp in enumerate(np.arange(0,3.55,0.05)):
    A = fft2(a)
    kx,ky = np.meshgrid(fftfreq(a.shape[0]), fftfreq(a.shape[1]))
    K = (kx**2 + ky**2)**0.5
    A_ = A / (K+0.0001)**(exp)
    a_ = ifft2(A_)
    a_ = (a_ - np.mean(a_))/np.std(a_)
    
    
    fig, ax = plt.subplots()
    fig.set_size_inches(9,7)
    c = ax.imshow(np.real(a_), cmap='YlGnBu', vmin=-3,vmax=3)
    ax.text(0.02, 0.02, r'$\alpha=%.2f$'%exp, transform=ax.transAxes)
    fig.colorbar(c)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig('im_%03d.jpg'%(i+1),dpi=200)
    plt.close()