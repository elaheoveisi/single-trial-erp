import os
import matplotlib.pyplot as plt
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
import numpy as np
import pywt

# EEG Load + Preprocess
def load_eeg_data(filepath, selected_channels=None):
    """Load and preprocess EEG data from a BDF file."""
    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")
    
    raw = mne.io.read_raw_bdf(filepath, preload=True)
    print(f"Channels in BDF file: {raw.ch_names}, total: {len(raw.ch_names)}")
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="ignore")
    raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])
    print(f"Channels after montage: {raw.ch_names}, total: {len(raw.ch_names)}")

    # Select channels if specified
    if selected_channels:
        selected_available = [ch for ch in selected_channels if ch in raw.ch_names]
        raw.pick_channels(selected_available)
        print(f"Selected EEG channels: {selected_available}, total: {len(selected_available)}")

    # Apply bandpass filter
    raw.filter(l_freq=1.0, h_freq=60.0)
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    print(f"Number of channels in epochs: {len(epochs.ch_names)}")
    return raw, epochs

# Clean EEG
def clean_eeg(subject_id, task_id, config=None, save_dir=None):
    """Clean EEG data for a given subject and task."""
    filepath = f"{config['target_dir']}/sub-{subject_id}/eeg/sub-{subject_id}_task-{task_id}_eeg.bdf"
    if not os.path.exists(filepath):
        raise ValueError(f"No file found for subject {subject_id}, task {task_id}")
    print(f"Processing: {filepath}")

    selected_channels = config.get("selected_channels", None)
    raw, epochs = load_eeg_data(filepath, selected_channels=selected_channels)

    # Rereferencing
    raw.set_eeg_reference(ref_channels="average", projection=True)
    epochs.set_eeg_reference(ref_channels="average", projection=True)

    # AutoReject for artifact rejection
    ar = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=True)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    # Access bad channels from raw.info['bads']
    bad_channels = raw.info['bads'] if raw.info['bads'] else []
    print("Bad channels identified:", bad_channels)

    # Interpolate bad channels if any
    if bad_channels:
        raw.interpolate_bads()
        print("Bad channels interpolated")
    else:
        print("No bad channels to interpolate")

    # Check for bad epochs
    bad_epoch_indices = np.where(reject_log.bad_epochs)[0]
    if len(bad_epoch_indices) > 0:
        print(f"{len(bad_epoch_indices)} bad epochs found.")
        fig_bad = epochs[bad_epoch_indices].plot(
            scalings=dict(eeg=100e-6), show=False
        )
        fig_bad.suptitle(f"Bad Epochs - Subject {subject_id} Task {task_id}")

        save_dir = config["save_plot_dir"]
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(
            save_dir,
            f"sub-{subject_id}_task-{task_id}_bad_epochs.png"
        )
        try:
            fig_bad.savefig(fig_path)
            print(f"Saved bad epoch plot to {fig_path}")
        except PermissionError:
            print(f"Permission denied: Cannot save plot to '{fig_path}'. Please check folder permissions.")
        plt.show()
    else:
        print("No bad epochs found.")

    # Plot reject log
    reject_log.plot("horizontal")

    # ICA processing
    n_channels = len(epochs_clean.ch_names)
    n_components = min(20, n_channels)  # Use the smaller of 20 or the number of channels
    print(f"Using {n_components} ICA components for {n_channels} channels")
    ica = ICA(n_components=n_components, method="picard", random_state=0)
    print("Initial ICA exclude list:", ica.exclude)
    ica.fit(epochs_clean)
    ica.plot_components(inst=raw)
    plt.suptitle(f"ICA Components - Subject {subject_id} Task {task_id}")
    plt.show()
    print("ICA exclude list after fitting:", ica.exclude)

    ica.plot_overlay(epochs_clean.average(), exclude=ica.exclude)
    plt.suptitle(f"ICA Overlay - Subject {subject_id} Task {task_id}")
    plt.show()

    # Wavelet denoising of ICA components
    sources = ica.get_sources(epochs_clean).get_data()
    cleaned_sources = np.zeros_like(sources)
    wavelet = 'coif5'
    level = 5

    for epoch_idx in range(sources.shape[0]):
        for comp_idx in range(sources.shape[1]):
            signal = sources[epoch_idx, comp_idx, :]
            max_level = pywt.swt_max_level(len(signal))
            current_level = min(level, max_level)
            coeffs = pywt.swt(signal, wavelet=wavelet, level=current_level)
            thresholded_coeffs = []
            for cA, cD in coeffs:
                threshold = np.median(np.abs(cD)) / 0.6745 * np.sqrt(2 * np.log(len(cD)))
                cD_thresh = pywt.threshold(cD, threshold, mode='soft')
                thresholded_coeffs.append((cA, cD_thresh))
            cleaned_signal = pywt.iswt(thresholded_coeffs, wavelet=wavelet)
            cleaned_sources[epoch_idx, comp_idx, :] = cleaned_signal

    ica_sources = ica.get_sources(epochs_clean)
    ica_sources._data = cleaned_sources

    # Apply ICA to clean data
    raw_clean = ica.apply(raw.copy())
    ica.apply(epochs_clean)

    # Plot cleaned raw data
    raw_clean.plot(
        scalings="auto", title=f"Cleaned Raw - Subject {subject_id} Task {task_id}"
    )
    plt.show()

    # Plot evoked response
    evoked = epochs_clean.average()
    print("Evoked shape:", evoked.data.shape)
    evoked._data *= 1e6  # Convert to microvolts for plotting
    fig_evoked = evoked.plot(scalings=dict(eeg=20), time_unit="s", show=False)
    fig_evoked.suptitle(f"Evoked - Subject {subject_id} Task {task_id}")
    if config["save_plot_dir"]:
        fig_path = os.path.join(
            config["save_plot_dir"], f"sub-{subject_id}_task-{task_id}_evoked.png"
        )
        try:
            fig_evoked.savefig(fig_path)
            print(f"Saved evoked plot to {fig_path}")
        except PermissionError:
            print(f"Permission denied: Cannot save plot to '{fig_path}'. Please check folder permissions.")
    plt.show()

    # Compare bad vs clean epochs if bad epochs exist
    if len(bad_epoch_indices) > 0:
        evoked_bad = epochs[bad_epoch_indices].average()
        plt.figure()
        plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, "r", zorder=-1)
        evoked_clean = epochs_clean.average()
        evoked_clean.plot(axes=plt.gca(), show=False)
        plt.title(f"Bad vs Clean - Subject {subject_id} Task {task_id}")
        if config["save_plot_dir"]:
            fig_path = os.path.join(
                config["save_plot_dir"],
                f"sub-{subject_id}_task-{task_id}_bad_vs_clean.png"
            )
            try:
                plt.savefig(fig_path)
                print(f"Saved bad vs clean plot to {fig_path}")
            except PermissionError:
                print(f"Permission denied: Cannot save plot to '{fig_path}'. Please check folder permissions.")
        plt.show()

    # Store bad channels in the cleaned data
    raw_clean.info['bads'] = bad_channels
    epochs_clean.info['bads'] = bad_channels

    return raw_clean, epochs_clean

# Preprocess all subjects and tasks
def preprocess_all_subjects_tasks(subjects, tasks, config=None, save_dir=None):
    """Preprocess EEG data for all subjects and tasks."""
    for subject_id in subjects:
        for task_id in tasks:
            raw_clean, epochs_clean = clean_eeg(
                subject_id, task_id, config=config
            )
            path = f"{config['processed_dir']}/sub-{subject_id}_task-{task_id}_cleaned_raw.fif"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                raw_clean.save(path, overwrite=True)
                print(f"Saved cleaned data to: {path}")
            except PermissionError:
                print(f"Permission denied: Cannot save file to '{path}'. Please check folder permissions.")