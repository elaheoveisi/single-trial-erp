import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from autoreject import AutoReject
import os
import autoreject
import matplotlib.pyplot as plt
import openneuro

# === File map ===
dataset = 'ds005841'   # The id code on OpenNeuro for this example dataset
subject_id = '001'    
tasks = ['lumfront', 'lumperp']

target_dir = os.path.join(
    os.path.dirname(autoreject.__file__), '..', 'examples', dataset)
os.makedirs(target_dir, exist_ok=True)

# === Download only the needed files for each task ===
include_paths = [f'sub-{subject_id}/eeg/sub-{subject_id}_task-{task}_eeg.bdf' for task in tasks]

openneuro.download(dataset=dataset, target_dir=target_dir, include=include_paths)

#file_map
file_map = {}
for task in tasks:
    bdf_path = os.path.join(target_dir, f"sub-{subject_id}", "eeg", f"sub-{subject_id}_task-{task}_eeg.bdf")
    file_map[(int(subject_id), task)] = bdf_path




# === Load and preprocess raw EEG ===
def load_and_preprocess(raw_fname):
    """
    Load and preprocess raw EEG data from a BDF file.

    Parameters
    ----------
    raw_fname : str
        Path to the raw EEG BDF file.

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw EEG data.
    epochs : mne.Epochs
        Fixed-length epochs created from the raw EEG.
    """
    raw = mne.io.read_raw_bdf(raw_fname, preload=True)
    
    # Set standard BioSemi 64 montage
    montage = mne.channels.make_standard_montage('biosemi64')
    raw.set_montage(montage, on_missing='ignore')
    
    # Keep only EEG channels (drop EXG, status, etc.)
    raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])

    # Apply bandpass filter
    raw.filter(l_freq=1.0, h_freq=60.0)

    # Create fixed-length epochs after filtering
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)

    return raw, epochs


def autoreject_and_ica(raw, epochs):
    """
    Apply autoreject and ICA to the raw EEG data and epochs.

    Parameters
    ----------
    raw : mne.io.Raw
        The preprocessed raw EEG data.
    epochs : mne.Epochs
        Epochs created from raw EEG.

    Returns
    -------
    raw_clean : mne.io.Raw
        ICA-cleaned raw EEG data.
    epochs : mne.Epochs
        Original epochs.
    epochs_clean : mne.Epochs
        Cleaned epochs after autoreject and ICA.
    reject_log : autoreject.RejectLog
        Log of rejected epochs.
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    """
    # === Apply AutoReject ===
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                               n_jobs=1, verbose=True)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)

    # === Plot bad epochs if any ===
    if len(reject_log.bad_epochs) > 0:
        print(f"{len(reject_log.bad_epochs)} bad epochs found.")
        epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6), title="Rejected Epochs")
    else:
        print("No bad epochs found.")

    # === ICA decomposition ===
    ica = ICA(n_components=64, method='picard', random_state=0)
    ica.fit(epochs_clean)

    # === Plot ICA components ===
    ica.plot_components(inst=raw)
    plt.show()

    # === Overlay ICA on averaged epochs ===
    ica.plot_overlay(epochs.average(), exclude=ica.exclude)
    plt.show()

    # === Apply ICA to raw and epochs ===
    ica.exclude = []  # (you can set specific components if needed)
    raw_clean = ica.apply(raw.copy())
    ica.apply(epochs, exclude=ica.exclude)

    return raw_clean, epochs, epochs_clean, reject_log, ica



# === Visualization of raw + evoked plots ===
def visualize(raw, epochs_clean):
    """
    Visualize the raw EEG data and averaged evoked response.

    Parameters
    ----------
    raw : mne.io.Raw
        Cleaned raw EEG data.
    epochs_clean : mne.Epochs
        Cleaned epochs used for averaging and plotting.

    Returns
    -------
    None
    """
    raw.plot(scalings='auto')
    plt.show()

    evoked = epochs_clean.average()
    print("Evoked data shape:", evoked.data.shape)
    evoked._data *= 1e6  # Convert to microvolts
    evoked.plot(scalings=dict(eeg=64), time_unit='s')
    plt.show()


# === Final wrapper in your required format ===
def clean_eeg(subject_id, task_id):
    """
    Wrapper to load, clean, and visualize EEG data for a given subject and task.

    Parameters
    ----------
    subject_id : int
        The subject identifier.
    task_id : str
        The task name (e.g., "lumfront", "lumperp").

    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned raw EEG data.
    epochs_clean : mne.Epochs
        Cleaned epochs after autoreject.
    """
    filepath = file_map.get((subject_id, task_id))
    if not filepath:
        raise ValueError(f"No file found for subject {subject_id}, task {task_id}")

    print(f"Processing: {filepath}")
    
    # Load and preprocess EEG → returns raw and epochs
    raw, epochs = load_and_preprocess(filepath)

    # Run autoreject and ICA
    raw_clean, epochs, epochs_clean, reject_log, ica = autoreject_and_ica(raw, epochs)

    # Visualize results
    visualize(raw_clean, epochs_clean)

    #Plot ICA overlay
    ica.plot_overlay(raw, exclude=ica.exclude)
    plt.show()

    # Plot bad epochs if any
    if len(reject_log.bad_epochs) > 0:
        evoked_bad = epochs[reject_log.bad_epochs].average()
        plt.figure()
        plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
        epochs_clean.average().plot(axes=plt.gca())
        plt.show()

    return raw_clean, epochs_clean



# === Run all subjects and tasks ===
if __name__ == "__main__":
    subjects = [1]  
    tasks = ["lumfront", "lumperp"]

    save_dir = r"C:\Users\elahe\Documents\EEG_Cleaned"
    os.makedirs(save_dir, exist_ok=True)

    for subject_id in subjects:
        for task_id in tasks:
            raw_clean, epochs_clean = clean_eeg(subject_id, task_id)

            # === Save after each subject-task
            filename = f"sub-{subject_id}_task-{task_id}_cleaned_raw.fif"
            save_path = os.path.join(save_dir, filename)
            raw_clean.save(save_path, overwrite=True)
            print(f"Saved cleaned data to: {save_path}")