# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal
import numpy.linalg

def stoi(x, y, fs_signal):
    #   d = stoi(x, y, fs_signal) returns the output of the short-time
    #   objective intelligibility (STOI) measure described in [1, 2], where x 
    #   and y denote the clean and processed speech, respectively, with sample
    #   rate fs_signal in Hz. The output d is expected to have a monotonic 
    #   relation with the subjective speech-intelligibility, where a higher d 
    #   denotes better intelligible speech. See [1, 2] for more details.
    #
    #   References:
    #      [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
    #      Objective Intelligibility Measure for Time-Frequency Weighted Noisy
    #      Speech', ICASSP 2010, Texas, Dallas.
    #
    #      [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for 
    #      Intelligibility Prediction of Time-Frequency Weighted Noisy Speech', 
    #      IEEE Transactions on Audio, Speech, and Language Processing, 2011. 

    if len(x) != len(y):
        raise ValueError('x and y should have the same length')

    # initialization
    x = x[:]   # clean speech column vector
    y = y[:]   # processed speech column vector

    fs = 10000.0    # sample rate of proposed intelligibility measure
    N_frame = 256   # window support
    K = 512   # FFT size
    J = 15    # Number of 1/3 octave bands
    mn = 150    # Center frequency of first 1/3 octave band in Hz.
    H = thirdoct(fs,K,J,mn)   # Get 1/3 octave band matrix
    N = 30   # Number of frames for intermediate intelligibility measure (Length analysis window)
    Beta = -15    # lower SDR-bound
    dyn_range = 40   # speech dynamic range

    # resample signals if other samplerate is used than fs
    if int(fs_signal) != int(fs):
        x	= scipy.signal.resample_poly(x,fs,fs_signal)
        y 	= scipy.signal.resample_poly(y,fs,fs_signal)

    # remove silent frames
    x, y = removeSilentFrames(x,y,dyn_range,N_frame,N_frame//2)

    # apply 1/3 octave band TF-decomposition
    x_hat = stdft(x,N_frame,N_frame//2,K)  # apply short-time DFT to clean speech
    y_hat	= stdft(y,N_frame,N_frame//2,K)  # apply short-time DFT to processed speech

    x_hat = x_hat[:,0:K//2+1].T  # take clean single-sided spectrum
    y_hat = y_hat[:,0:K//2+1].T  # take processed single-sided spectrum
    
    N_bin, nframes = x_hat.shape
    
    X = np.zeros((J,nframes))  # init memory for clean speech 1/3 octave band TF-representation 
    Y = np.zeros((J,nframes))  # init memory for processed speech 1/3 octave band TF-representation 

    for i in range(nframes):
        X[:,i] = np.sqrt(np.dot(H,np.abs(x_hat[:,i])**2)) # apply 1/3 octave bands as described in Eq.(1) [1]
        Y[:,i] = np.sqrt(np.dot(H,np.abs(y_hat[:,i])**2))

    # loop al segments of length N and obtain intermediate intelligibility measure for all TF-regions
    d_interm = np.zeros((J,nframes-N+1))  # init memory for intermediate intelligibility measure
    c = 10**(-Beta/20.0)  # constant for clipping procedure

    for m in range(N,nframes+1):
        X_seg = X[:,m-N:m] # region with length N of clean TF-units for all j
        Y_seg = Y[:,m-N:m] # region with length N of processed TF-units for all j
        alpha = np.sqrt(np.sum(X_seg**2,axis=1)/np.sum(Y_seg**2,axis=1))   # obtain scale factor for normalizing processed TF-region for all j
        aY_seg = Y_seg*np.tile(np.array([alpha]).T,(1,N)) # obtain \alpha*Y_j(n) from Eq.(2) [1]
        for j in range(J):
            Y_prime = np.min(np.vstack((aY_seg[j,:],X_seg[j,:]+X_seg[j,:]*c)),axis=0) # apply clipping from Eq.(3)   	
            d_interm[j,m-N] = taa_corr(X_seg[j,:],Y_prime[:]) # obtain correlation coeffecient from Eq.(4) [1]
    
    d = np.mean(d_interm[:]) # combine all intermediate intelligibility measures as in Eq.(4) [1]
    
    return d
    


def thirdoct(fs,N_fft,numBands,mn):
    #   [A CF] = THIRDOCT(FS, N_FFT, NUMBANDS, MN) returns 1/3 octave band matrix
    #   inputs:
    #       FS:         samplerate 
    #       N_FFT:      FFT size
    #       NUMBANDS:   number of bands
    #       MN:         center frequency of first 1/3 octave band
    #   outputs:
    #       A:          octave band matrix
    #       CF:         center frequencies

    f = np.linspace(0.0,fs,N_fft+1)
    f = f[0:N_fft//2+1]
    k = np.arange(0.0,numBands)
    cf = 2**(k/3.0)*mn
    fl = np.sqrt((2**(k/3.0)*mn)*2**((k-1)/3.0)*mn)
    fr = np.sqrt((2**(k/3.0)*mn)*2**((k+1)/3.0)*mn)
    A = np.zeros((numBands,len(f)))

    for i in range(len(cf)):
        b = np.argmin((f-fl[i])**2)
        fl[i] = f[b]
        fl_ii = b
        
        b = np.argmin((f-fr[i])**2)
        fr[i] = f[b]
        fr_ii = b
        A[i,fl_ii:fr_ii] = 1

    rnk = np.sum(A,axis=1)
    
    numBands = None
    for i in range(len(rnk)-1,0,-1):
        if rnk[i] >= rnk[i-1] and rnk[i] != 0:
            numBands = i + 1
            break
    if numBands is None:
        raise ValueError('numBands cannot be None.')
        
    A = A[0:numBands,:]
    cf = cf[0:numBands]
    
    return A
    


def stdft(x,N,K,N_fft):
    #   X_STDFT = X_STDFT(X, N, K, N_FFT) returns the short-time
    #   hanning-windowed dft of X with frame-size N, overlap K and DFT size
    #   N_FFT. The columns and rows of X_STDFT denote the frame-index and
    #   dft-bin index, respectively.

    frames = range(0,len(x)-N,K)
    x_stdft = np.zeros((len(frames),N_fft),dtype=complex)

    w = np.hanning(N+2)
    w = w[1:-1]
    
    x = x[:]

    for i in range(len(frames)):
        x_stdft[i,:] = np.fft.fft(x[frames[i]:frames[i]+N]*w,N_fft)
        
    return x_stdft


def removeSilentFrames(x,y,Range,N,K):
    #   [X_SIL Y_SIL] = REMOVESILENTFRAMES(X, Y, RANGE, N, K) X and Y
    #   are segmented with frame-length N and overlap K, where the maximum energy
    #   of all frames of X is determined, say X_MAX. X_SIL and Y_SIL are the
    #   reconstructed signals, excluding the frames, where the energy of a frame
    #   of X is smaller than X_MAX-RANGE
    
    x = x[:]
    y = y[:]

    frames = range(0,len(x)-N,K)
    w = np.hanning(N+2)
    w = w[1:-1]
    
    msk = np.zeros(len(frames))

    for j in range(len(frames)):
        jj = range(frames[j],frames[j]+N)
        msk[j] = 20.0*np.log10(numpy.linalg.norm(x[frames[j]:frames[j]+N]*w)/np.sqrt(N)+1e-100)

    msk = (msk-max(msk)+Range)>0
    count = 0

    x_sil = np.zeros(len(x))
    y_sil = np.zeros(len(y))

    for j in range(len(frames)):
        if msk[j]:
            x_sil[frames[count]:frames[count]+N] += x[frames[j]:frames[j]+N]*w
            y_sil[frames[count]:frames[count]+N] += y[frames[j]:frames[j]+N]*w
            count += 1
            
    x_sil = x_sil[0:frames[count-1]+N]
    y_sil = y_sil[0:frames[count-1]+N]
    
    return x_sil, y_sil


def taa_corr(x,y):
    #   RHO = TAA_CORR(X, Y) Returns correlation coeffecient between column
    #   vectors x and y. Gives same results as 'corr' from statistics toolbox.
    xn = x - np.mean(x)
    xn = xn / np.sqrt(np.sum(xn**2))
    yn = y - np.mean(y)
    yn = yn / np.sqrt(np.sum(yn**2))
    rho = np.sum(xn*yn)
    
    return rho
    
