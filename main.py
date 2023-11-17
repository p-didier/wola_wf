# Purpose of script:
# Example of WOLA-based filtering of a single-channel noisy signal.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created: 17/11/2023

import sys
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path

FS = 16000  # sampling frequency [samples/s]
DUR = 5  # signal duration [s]
N_DFT = 1024  # DFT size
OVLP = 0.5  # WOLA window overlap
WINDOW = np.sqrt(np.hanning(N_DFT))  # window function

BETA = 0.9  # forgetting factor for exponential averaging of PSDs

DESIRED_SIGNAL_TYPE = 'speech'  # 'speech' or 'noise'
# DESIRED_SIGNAL_TYPE = 'noise'  # 'speech' or 'noise'
SNR = 0  # desired signal-to-noise ratio [dB]

# Speech activity detection mechanism
SPEECH_ACTIVITY_DETECTOR = 'VAD'  # 'VAD' or 'SPP'
# SPEECH_ACTIVITY_DETECTOR = 'SPP'  # 'VAD' or 'SPP'

SPEECH_FILE = f'{Path(__file__).parent}\\speech_example.wav'


# Parameters checks
if DESIRED_SIGNAL_TYPE == 'noise' and SPEECH_ACTIVITY_DETECTOR == 'SPP':
    print('WARNING: SPP-based speech activity detection is not valid for noise signals. Using VAD instead.')
    SPEECH_ACTIVITY_DETECTOR = 'VAD'

def main():
    """Main function (called by default when running script)."""
    # Get microphone signal
    y, activity = get_signal(plotit=False)
    # Go to WOLA domain
    yWOLA, f, t = get_stft(y, FS, WINDOW, OVLP)
    # Loop over time frames
    Ryy, Rnn = np.zeros(len(f)), np.zeros(len(f))
    Ns = int(N_DFT * (1 - OVLP))
    outWOLA = np.zeros_like(yWOLA)
    nUpRyy, nUpRnn = np.zeros(len(f)), np.zeros(len(f))
    for l in range(len(t)):
        print(f'Frame {l+1}/{len(t)}')
        # Loop over frequency bins
        for kappa in range(len(f)):
            # Determine frame activity
            if SPEECH_ACTIVITY_DETECTOR == 'VAD':
                currActivity = np.sum(activity[l * Ns : (l + 1) * Ns]) > 0.5 * Ns  # VAD is on iff >50% of frame is voiced
            elif SPEECH_ACTIVITY_DETECTOR == 'SPP':
                currActivity = activity[kappa, l] > 0.5  # there is activity iff SPP > 0.5
                
            if currActivity:
                Ryy[kappa] = BETA * Ryy[kappa] + (1 - BETA) * np.abs(yWOLA[kappa, l])**2
                nUpRyy[kappa] += 1
            else:
                Rnn[kappa] = BETA * Rnn[kappa] + (1 - BETA) * np.abs(yWOLA[kappa, l])**2
                nUpRnn[kappa] += 1
            # Compute Wiener filter
            if nUpRyy[kappa] > 0 and nUpRnn[kappa] > 0:
                wf = 1 / Ryy[kappa] * (Ryy[kappa] - Rnn[kappa])
            else:
                wf = 1  # no update yet, so no filtering
            # Apply Wiener filter
            outWOLA[kappa, l] = wf * yWOLA[kappa, l]
    # Go back to time domain
    _, out = sig.istft(
        outWOLA,
        fs=FS,
        window=WINDOW,
        nperseg=len(WINDOW),
        noverlap=int(OVLP * len(WINDOW)),
        boundary='zeros'
    )

    subfolder = f'{Path(__file__).parent}\\exports\\using{SPEECH_ACTIVITY_DETECTOR}'
    if not Path(subfolder).exists():
        Path(subfolder).mkdir()
    # Plot results
    fig1, fig2 = plots(y, out, yWOLA, outWOLA, t, f)
    fig1.savefig(f'{subfolder}\\time_domain.png', dpi=300)
    fig2.savefig(f'{subfolder}\\wola_domain.png', dpi=300)
    # Export as WAV
    sf.write(f'{subfolder}\\in.wav', y, FS)
    sf.write(f'{subfolder}\\out.wav', out, FS)


def plots(y, out, yWOLA, outWOLA, t, f):
    # Time-domain signals
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(6.5, 3.5)
    axes.plot(y, label='Noisy signal $y = s + n$')
    axes.plot(out, label='Filtered signal $y_{\mathrm{out}}$')
    axes.grid()
    axes.legend()
    axes.set_xlabel('Time [samples]')
    axes.set_ylabel('Amplitude')
    axes.set_title(f'Parameters: $N_{{\mathrm{{DFT}}}} = {N_DFT}$, $N_{{\mathrm{{OVL}}}} = {OVLP}$, $\\beta = {BETA}$, $\\mathrm{{SNR}} = {SNR}$ dB')
    fig.tight_layout()
    plt.show(block=False)

    # WOLA-domain signals
    fig2, axes = plt.subplots(2,1)
    fig2.set_size_inches(6.5, 4)
    # Determine colorbar limits
    vmin = np.amax([-100, np.amin([np.nanmin(20 * np.log10(np.abs(yWOLA))), np.nanmin(20 * np.log10(np.abs(outWOLA)))])])
    vmax = np.amin([0, np.amax([np.nanmax(20 * np.log10(np.abs(yWOLA))), np.nanmax(20 * np.log10(np.abs(outWOLA)))])])
    mapp = axes[0].pcolormesh(t, f, 20 * np.log10(np.abs(yWOLA)), shading='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Noisy signal $y = s + n$')
    axes[0].set_ylabel('Frequency [Hz]')
    cb = fig2.colorbar(mapp, ax=axes[0])
    cb.set_label('Magnitude [dB]')
    mapp = axes[1].pcolormesh(t, f, 20 * np.log10(np.abs(outWOLA)), shading='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('Filtered signal $y_{\mathrm{out}}$')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Frequency [Hz]')
    cb = fig2.colorbar(mapp, ax=axes[1])
    cb.set_label('Magnitude [dB]')
    fig2.tight_layout()
    axes[0].set_title(f'Parameters: $N_{{\mathrm{{DFT}}}} = {N_DFT}$, $N_{{\mathrm{{OVL}}}} = {OVLP}$, $\\beta = {BETA}$, $\\mathrm{{SNR}} = {SNR}$ dB')
    plt.show(block=False)

    return fig, fig2


def get_signal(plotit=False):
    """Get desired signal and add noise to it. Possibly plot."""
    s, activity = get_des_sig()  # get desired signal
    n = np.random.randn(int(FS * DUR))
    # Apply correct SNR
    n *= np.sqrt(np.sum(s**2) / np.sum(n**2)) / (10**(SNR / 20))
    y = s + n

    if plotit:  # plot signals
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(8.5, 3.5)
        axes.plot(y, label='Noisy signal $y = s + n$')
        axes.plot(s, label=f'Desired signal $s$ (type: {DESIRED_SIGNAL_TYPE})')
        axes.plot(n, label='Noise $n$')
        axes.grid()
        axes.legend()
        axes.set_xlabel('Time [samples]')
        axes.set_ylabel('Amplitude')
        fig.tight_layout()
        plt.show()
    
    return y, activity


def get_des_sig():
    if DESIRED_SIGNAL_TYPE == 'speech':
        signal, fs = sf.read(SPEECH_FILE)
        # Resample if necessary
        if fs != FS:
            signal = sig.resample(signal, int(len(signal) * FS / fs))
        signal = signal[:int(FS * DUR)]
        if SPEECH_ACTIVITY_DETECTOR == 'VAD':
            # Compute energy-based VAD
            activity, _ = oracleVAD(signal, 0.025, 1e-3, fs)  # 25 ms window, 1e-3 energy threshold (HARD-CODED parameters, can be adjusted)
        elif SPEECH_ACTIVITY_DETECTOR == 'SPP':
            # Compute SPP
            Signal = get_stft(signal, fs, WINDOW, OVLP)[0]
            activity = oracleSPP(Signal, plotSPP=False)
    elif DESIRED_SIGNAL_TYPE == 'noise':
        baseSig = np.random.randn(int(FS * DUR))
        # Add random pauses with a duration multiple of `N_DFT` and separated by
        # other multiples `N_DFT` samples.
        pauseMask = np.array([])
        while len(pauseMask) < len(baseSig):
            pauseMask = np.concatenate((
                pauseMask,
                np.zeros(N_DFT * np.random.randint(1, 5)),
                np.ones(N_DFT * np.random.randint(1, 5))
            ))
        pauseMask = pauseMask[:len(baseSig)]
        signal = baseSig * pauseMask
        activity = pauseMask.astype(bool)
    
    return signal, activity


def get_stft(x: np.ndarray, fs, win, ovlp, boundary='zeros'):
    """Compute STFT representation of a single-channel time-domain signal."""
    f, t, out = sig.stft(
        x,
        fs=fs,
        window=win,
        nperseg=len(win),
        noverlap=int(ovlp * len(win)),
        return_onesided=True,
        boundary=boundary
    )
    return out, f, t


def oracleVAD(x, tw, thrs, Fs):
    """
    Oracle Voice Activity Detection (VAD) function. Returns the
    oracle VAD for a given speech (+ background noise) signal <x>.
    Based on the computation of the short-time signal energy.
    
    Parameters
    ----------
    -x [N*1 float vector, -] - Time-domain signal.
    -tw [float, s] - VAD window length.
    -thrs [float, [<x>]^2] - Energy threshold.
    -Fs [int, samples/s] - Sampling frequency.
    
    Returns
    -------
    -oVAD [N*1 binary vector] - Oracle VAD corresponding to <x>.

    (c) Paul Didier - 13-Sept-2021
    SOUNDS ETN - KU Leuven ESAT STADIUS
    ------------------------------------
    """

    # Check input format
    x = np.array(x)     # Ensure it is an array
    if len(x.shape) > 1:
        print('<oracleVAD>: input signal is multidimensional: \
            using 1st row as reference')
        dimsidx = range(len(x.shape))
        # Rearrange x dimensions in increasing order of size
        x = np.transpose(x, tuple(np.take(dimsidx,np.argsort(x.shape))))
        for ii in range(x.ndim-1):
            x = x[0]    # extract 1 "row" along the largest dimension

    # Number of samples
    n = len(x)

    # VAD window length
    if tw > 0:
        Nw = int(tw * Fs)
    else:
        Nw = 1

    # Compute VAD
    oVAD = np.zeros(n)
    for ii in range(n):
        # Extract chunk
        idxBeg = int(np.amax([ii - Nw // 2, 0]))
        idxEnd = int(np.amin([ii + Nw // 2, len(x)]))
        # Compute VAD frame
        oVAD[ii] = compute_VAD(x[idxBeg:idxEnd], thrs)

    # Time vector
    t = np.arange(n)/Fs

    return oVAD,t


# @njit  # <-- possible JIT compilation for speed (numba package)
def compute_VAD(chunk_x,thrs):
    # Compute short-term signal energy
    energy = np.mean(np.abs(chunk_x)**2)
    # Assign VAD value
    if energy > thrs:
        VADout = 1
    else:
        VADout = 0
    return VADout


def oracleSPP(X, plotSPP=False):
    """Oracle Speech Presence Probability (SPP) computation.
    Returns the oracle SPP for a given speech STFT <X>.
    --- Based on implementation by R.Ali (2018)
        (itself based on implementation by T.Gerkmann (2011)).

    Parameters
    ----------
    - X : [Nf * Nt] complex array
        Signal STFT (onesided, freqbins x time frames) - SINGLE CHANNEL.
    - thrs : float
        Energy threshold.
    - plotSPP : bool
        If true, plots function output on a figure.
    
    Returns
    -------
    - SPPout : [Nf * Nt] float array
        Oracle SPP corresponding to <X>.

    (c) Paul Didier - 6-Oct-2021
    SOUNDS ETN - KU Leuven ESAT STADIUS
    """

    # --------------------------------
    # NB: many parameters are hard-coded in this function
    # --------------------------------

    # Useful constants
    nFrames = X.shape[1]

    # Allocate memory
    noisePowMat = np.zeros_like(X)
    SPPout = np.zeros_like(X, dtype=float)

    # Compute initial noise PSD estimate --> Assumes that the first 5 time-frames are noise-only.
    noise_psd_init = np.mean(np.abs(X[:,:5])**2, 1)
    noisePowMat[:, 0] = noise_psd_init

    # Parmeters
    PH1mean  = 0.5
    alphaPH1mean = 0.5
    alphaPSD = 0.8

    # constants for a posteriori SPP
    q          = 0.5                    # a priori probability of speech presence
    priorFact  = q/(1 - q)
    xiOptDb    = 15                     # optimal fixed a priori SNR for SPP estimation
    xiOpt      = 10**(xiOptDb/10)
    logGLRFact = np.log(1/(1 + xiOpt))
    GLRexp     = xiOpt/(1 + xiOpt)

    for indFr in range(nFrames):  # All frequencies are kept in a vector

        noisyDftFrame = X[:,indFr]
        noisyPer = noisyDftFrame * noisyDftFrame.conj()
        snrPost1 =  noisyPer / noise_psd_init  # a posteriori SNR based on old noise power estimate

        # Noise power estimation
        inside_exp = logGLRFact + GLRexp*snrPost1
        inside_exp[inside_exp > 200] = 200
        GLR     = priorFact * np.exp(inside_exp)
        PH1     = GLR/(1 + GLR) # a posteriori speech presence probability
        PH1mean  = alphaPH1mean * PH1mean + (1 - alphaPH1mean) * PH1
        tmp = PH1[PH1mean > 0.99]
        tmp[tmp > 0.99] = 0.99
        PH1[PH1mean > 0.99] = tmp
        estimate =  PH1 * noise_psd_init + (1 - PH1) * noisyPer 
        noise_psd_init = alphaPSD * noise_psd_init + (1 - alphaPSD) * estimate
        
        SPPout[:,indFr] = np.real(PH1)    
        noisePowMat[:,indFr] = noise_psd_init

    if plotSPP:
        fig, ax = plt.subplots()
        ax.imshow(SPPout)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        ax.set(title='SPP')
        ax.grid()
        plt.show()
    
    return SPPout


if __name__ == '__main__':
    sys.exit(main())