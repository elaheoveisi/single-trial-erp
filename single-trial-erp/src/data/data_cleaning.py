import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import time
import mne
from mne.channels import make_standard_montage
from mne.preprocessing import ICA
import numpy as np
import os

from pathlib import Path
import mne

def load_and_plot_raw_eeg(file_path=r"C:\Users\elahe\Downloads\sub-001_task-lumfront_eeg.bdf", n_channels=64):
    """
    Load and plot raw EEG data from a BDF file.

    Parameters:
    - file_path: str
        Path to the BDF file.
    - n_channels: int
        Number of channels to display in the plot.

    Returns:
    - raw: mne.io.Raw
        The loaded raw EEG object.
    """
    sample_data_raw_file = Path(file_path)

    # Load the BDF EEG data
    raw = mne.io.read_raw_bdf(sample_data_raw_file, preload=True)

    # Print summary and detailed info
    print(raw)
    print(raw.info)

    # Plot EEG data
    raw.plot(n_channels=n_channels, scalings='auto', title='Raw EEG Data', show=True)

    return raw

def set_montage(raw):
    """Apply a standard montage for channel locations, ignoring missing channels."""
    montage = make_standard_montage('standard_1005')  # 10-10 system for  72 channels
    raw.set_montage(montage, on_missing='ignore')  # Ignore unmapped EXG channels
    return raw



def load_and_plot_raw_bdf(file_path, n_channels=64, title='Raw EEG Data'):
    """
    Load BDF file and plot the first `n_channels` channels.
    """
    raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
    print("Initial Raw Info:")
    print(raw)
    print("Detailed Info:")
    print(raw.info)
    raw.plot(n_channels=n_channels, scalings='auto', title=title, show=True)
    print("Bad channels before cropping:", raw.info["bads"])
    return raw

def crop_and_replot(raw, crop_start=0, crop_end=300):
    """
    Crop bad channels
    """
    raw.crop(crop_start, crop_end).load_data()
    raw.plot(scalings='auto', title='Cropped EEG Data', show=True)
    print("Bad channels after cropping:", raw.info["bads"])
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
    raw  = load_and_plot_raw_eeg()
    plot_psd(raw)


def apply_filter(raw, l_freq=1, h_freq=40, fir_design="firwin"):
    """Apply bandpass filter to raw data."""
    raw.filter(l_freq, h_freq, fir_design=fir_design)
    return raw




def pick_eeg_channels(raw):
    """Select only EEG channels, excluding EXG."""
    raw.pick([ch for ch in raw.ch_names if not ch.startswith('EXG')])
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

if __name__ == "__main__":
    # Process data step by step
    raw = raw = load_and_plot_raw_eeg()
    raw = set_montage(raw)
    raw = pick_eeg_channels(raw)
    raw = apply_filter(raw)
    reject_criteria = get_reject_criteria()
    ica = run_ica(raw, reject=reject_criteria)








def extract_events_from_status(raw):
    """
    Extracts events from the ':Status' stimulus channel in the EEG recording.

    Parameters:
    - raw: mne.io.Raw
        The raw EEG data object.

    Returns:
    - raw: mne.io.Raw
        The raw object (with channel type updated if needed).
    - events: ndarray, shape (n_events, 3)
        The events array with sample index, previous value, and new value.
    - event_id: dict
        A dictionary mapping event labels to their corresponding integer codes.
    """
    # Make sure MNE treats ':Status' as a stim channel
    raw.set_channel_types({':Status': 'stim'})

    # Detect events from ':Status' channel
    events = mne.find_events(raw, stim_channel=':Status')

    # Define event code labels
    event_id = {
        "REF_LIGHT": 11,
        "REF_DARK": 12,
        "RAND_LIGHT": 13,
        "RAND_DARK": 14
    }

    print(f"Found {len(events)} events from ':Status' channel.")
    return raw, events, event_id





def create_epochs_and_plot_average(raw, events, event_id):
    """
    Create epochs, plot the average evoked response, and handle SSP projectors.

    Parameters:
        raw (mne.io.Raw): Filtered raw EEG data.
        events (ndarray): Events found in the raw data.
        event_id (dict): Mapping of event names to event codes.

    Returns:
        epochs (mne.Epochs): The epoched data.
        ssp_projectors (list): The list of SSP projectors from raw.info.
    """
    # Epoch the data
    epochs = mne.Epochs(raw, events, event_id=event_id, preload=True)

    # Plot evoked response for all channels with consistent axis limits
    plot_kwargs = dict(picks="all", ylim=dict(eeg=(-10, 10), eog=(-5, 15)))
    fig = epochs.average().plot(**plot_kwargs)
    fig.set_size_inches(6, 6)
    plt.show()


def extract_and_plot_eog_epochs(raw, baseline=(-0.5, -0.2)):
    """
    Extracts EOG epochs and plots EOG activity.

    Parameters:
        raw (mne.io.Raw): The preprocessed EEG data.
        baseline (tuple): Baseline correction interval (in seconds).

    Returns:
        eog_epochs (mne.Epochs): The epochs time-locked to EOG events.
    """
    # Create EOG epochs (detects blinks/saccades)
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=baseline)
    
    # Plot time x trial image of EOG activity
    eog_epochs.plot_image(combine="mean", title="EOG Epochs Image")

    # Plot average EOG response with topography
    eog_epochs.average().plot_joint(title="EOG Average Response")

    return eog_epochs