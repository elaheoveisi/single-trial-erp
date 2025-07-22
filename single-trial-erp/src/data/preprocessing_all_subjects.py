import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from autoreject import AutoReject
import os
import autoreject
import openneuro



# Files
dataset = 'ds005841'
subject_ids = ['001', '002']
tasks = ['lumfront', 'lumperp']

target_dir = os.path.join(
    os.path.dirname(autoreject.__file__), '..', 'examples', dataset)
os.makedirs(target_dir, exist_ok=True)

#  Build include paths 
include_paths = []
file_map = {}

for subj in subject_ids:
    for task in tasks:
        path = f'sub-{subj}/eeg/sub-{subj}_task-{task}_eeg.bdf'
        include_paths.append(path)

        # Also build the local path for file_map
        full_path = os.path.join(target_dir, f"sub-{subj}", "eeg", f"sub-{subj}_task-{task}_eeg.bdf")
        file_map[(int(subj), task)] = full_path

# Download from OpenNeuro 
openneuro.download(dataset=dataset, target_dir=target_dir, include=include_paths)

#  EEG cleaning function 
def clean_eeg(subject_id, task_id):
    """
    Load, clean (AutoReject + ICA), and visualize EEG data for a given subject and task.
    """
    filepath = file_map.get((subject_id, task_id))
    if not filepath:
        raise ValueError(f"No file found for subject {subject_id}, task {task_id}")
    print(f"Processing: {filepath}")

    # Load and preprocess raw EEG
    raw = mne.io.read_raw_bdf(filepath, preload=True)
    montage = mne.channels.make_standard_montage('biosemi64')
    raw.set_montage(montage, on_missing='ignore')
    raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])
    raw.filter(l_freq=1.0, h_freq=60.0)

    # Epoching
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)

    # AutoReject
    ar = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=True)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)

    if len(reject_log.bad_epochs) > 0:
        print(f"{len(reject_log.bad_epochs)} bad epochs found.")
        epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
        reject_log.plot('horizontal')

    # ICA
    ica = ICA(n_components=64, method='picard', random_state=0)
    ica.fit(epochs_clean)
    ica.plot_components(inst=raw)
    plt.show()
    ica.plot_overlay(epochs_clean.average(), exclude=ica.exclude)
    plt.show()

    ica.exclude = []  # Set manually 
    raw_clean = ica.apply(raw.copy())
    ica.apply(epochs_clean)

    # Final visualizations
    raw_clean.plot(scalings='auto')
    plt.show()
    evoked = epochs_clean.average()
    print("Evoked shape:", evoked.data.shape)
    evoked._data *= 1e6
    evoked.plot(scalings=dict(eeg=20), time_unit='s')
    plt.show()

    if len(reject_log.bad_epochs) > 0:
        evoked_bad = epochs[reject_log.bad_epochs].average()
        plt.figure()
        plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
        epochs_clean.average().plot(axes=plt.gca())
        plt.show()

    return raw_clean, epochs_clean

#  Run all subject-task pairs and save 
if __name__ == "__main__":
    subjects = [1]
    tasks = ['lumfront', 'lumperp']
    save_dir = r"C:\Users\elahe\Documents\EEG_Cleaned"
    os.makedirs(save_dir, exist_ok=True)

    for subject_id in subjects:
        for task_id in tasks:
            raw_clean, epochs_clean = clean_eeg(subject_id, task_id)
            filename = f"sub-{subject_id}_task-{task_id}_cleaned_raw.fif"
            save_path = os.path.join(save_dir, filename)
            raw_clean.save(save_path, overwrite=True)
            print(f"Saved cleaned data to: {save_path}")



