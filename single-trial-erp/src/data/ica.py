import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import time
import mne
from mne.channels import make_standard_montage
from mne.preprocessing import ICA

def load_raw_data(raw_fname=r"C:\Users\elahe\Downloads\sub-001_task-lumfront_eeg.bdf", tmin=0, tmax=60):
    """Load and crop raw EEG data."""
    raw = mne.io.read_raw_bdf(raw_fname, preload=True).crop(tmin, tmax)
    return raw

def set_montage(raw):
    """Apply a standard montage for channel locations, ignoring missing channels."""
    montage = make_standard_montage('standard_1005')  # 10-10 system for 72 channels
    raw.set_montage(montage, on_missing='ignore')  # Ignore unmapped EXG channels
    return raw

def pick_eeg_channels(raw):
    """Select only EEG channels, excluding EXG."""
    raw.pick([ch for ch in raw.ch_names if not ch.startswith('EXG')])
    return raw

def apply_filter(raw, l_freq=1, h_freq=40, fir_design="firwin"):
    """Apply bandpass filter to raw data."""
    raw.filter(l_freq, h_freq, fir_design=fir_design)
    return raw

def get_reject_criteria(threshold_eeg=150e-6):
    """Return dictionary for rejection criteria."""
    return dict(eeg=threshold_eeg)





def run_ica(raw, reject=None, n_components=20):
    """Fit ICA on raw data and plot components."""
    ica = ICA(n_components=n_components, method="picard", random_state=0)
    t0 = time()
    ica.fit(raw, reject=reject)
    fit_time = time() - t0
    title = f"ICA decomposition using picard (took {fit_time:.1f}s)"
    ica.plot_components(title=title)
    plt.show()
    return ica


def plot_possible_artifact_channels(raw):
    """
    Plot frontal EEG channels to inspect for eye/motion artifacts in absence of EOG/ECG.
    """
    possible_artifact_channels = [ch for ch in raw.ch_names if ch.startswith(('Fp', 'AF', 'Fz', 'FT'))]
    picks = mne.pick_channels(raw.ch_names, possible_artifact_channels)
    raw.plot(order=picks, n_channels=len(picks), show_scrollbars=False)



if __name__ == "__main__":
    # Process data step by step
    raw = load_raw_data()
    raw = set_montage(raw)
    raw = pick_eeg_channels(raw)
    raw = apply_filter(raw)
    plot_possible_artifact_channels(raw)
    reject_criteria = get_reject_criteria()
    ica = run_ica(raw, reject=reject_criteria)

