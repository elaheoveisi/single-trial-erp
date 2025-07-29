import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mne
import numpy as np

def load_raw_data(raw_fname=r"C:\Users\elahe\Downloads\sub-001_task-lumfront_eeg.bdf", tmin=0, tmax=60):
    """Load and crop raw EEG data."""
    raw = mne.io.read_raw_bdf(raw_fname, preload=True).crop(tmin, tmax)
    return raw

def plot_psd(raw, fmax=40):
    """
    Parameters:
    - raw: mne.io.Raw
        The raw EEG data object.
    - fmax: float, optional (default=250)# we regarded 40
        The maximum frequency to display in the PSD plot.

    Returns:
    - fig: matplotlib.figure.Figure
        The PSD figure object for further use or customization.
    """
    fig = raw.compute_psd(tmax=np.inf, fmax=fmax).plot(
        average=True,
        amplitude=False,
        picks="data",
        exclude="bads"
    )
    plt.show()
    return fig

if __name__ == "__main__":
    raw = load_raw_data()
    plot_psd(raw)



