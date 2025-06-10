from time import time

import mne
from mne.datasets import sample
from mne.preprocessing import ICA

print(__doc__)

data_path = sample.data_path()
eeg_path = data_path / "EEG" / "sample"
raw_fname = eeg_path / "sample_audvis_filt-0-40_raw.fif"

raw = mne.io.read_raw_fif(raw_fname).crop(0, 60).pick("eeg").load_data()

def load_preprocess_raw(raw_fname, crop_tmin=0, crop_tmax=60, l_freq=1, h_freq=40, fir_design="firwin"):
    """
    Load raw EEG data, crop, pick EEG channels, and apply bandpass filter.

    Returns:
    - raw : mne.io.Raw
    """
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.crop(crop_tmin, crop_tmax)
    raw.pick("eeg")
    raw.load_data()
    raw.filter(l_freq, h_freq, fir_design=fir_design)
    return raw

def get_reject_criteria(threshold_eeg=75e-6):
    """
    Return dictionary for rejection criteria.
    """
    return dict(eeg=threshold_eeg)




def run_ica(raw, method="picard", reject=None, n_components=20, fit_params=None, max_iter="auto", random_state=0):
    """
    Fit ICA on raw EEG data and plot the components.

    Returns:
    - ica : mne.preprocessing.ICA
    """
    ica = ICA(
        n_components=20,
        method=method,
        fit_params=fit_params,
        max_iter=auto,
        random_state=random_state,
    )
    t0 = time()
    ica.fit(raw, reject=reject)
    fit_time = time() - t0
    title = f"ICA decomposition using {method} (took {fit_time:.1f}s)"
    ica.plot_components(title=title)
    return ica

run_ica("picard")





