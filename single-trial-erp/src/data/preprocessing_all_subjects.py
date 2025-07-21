import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from autoreject import AutoReject

# === File map ===
file_map = {
    (1, "lumfront"): r"C:\Users\elahe\Downloads\sub-001_task-lumfront_eeg (2).bdf",
    (1, "lumperp"):  r"C:\Users\elahe\Downloads\sub-001_task-lumperp_eeg.bdf",
    }

# === Load and preprocess raw EEG ===
def load_and_preprocess(raw_fname):
    raw = mne.io.read_raw_bdf(raw_fname, preload=True)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')
    raw.pick([ch for ch in raw.ch_names if not ch.startswith('EXG')])
    raw.filter(l_freq=1.0, h_freq=60.0)
    return raw

# === Combined step: epoch + autoreject + ICA ===
def autoreject_and_ica(raw):
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    ar = AutoReject()
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    ica = ICA(n_components=20, method='picard', random_state=0)
    ica.fit(epochs_clean)
    ica.plot_components(inst=raw)
    plt.show()
    ica.exclude =[]
    raw_clean = ica.apply(raw.copy())

    return raw_clean, epochs, epochs_clean, reject_log, ica

# === Visualization of raw + evoked plots ===
def visualize(raw, epochs_clean):
    raw.plot(scalings='auto')
    plt.show()

    evoked = epochs_clean.average()
    print("Evoked data shape:", evoked.data.shape)
    evoked._data *= 1e6  # Convert to microvolts
    evoked.plot(scalings=dict(eeg=20), time_unit='s')
    plt.show()

# === Final wrapper in your required format ===
def clean_eeg(subject_id, task_id):
    filepath = file_map.get((subject_id, task_id))
    if not filepath:
        raise ValueError(f"No file found for subject {subject_id}, task {task_id}")

    print(f"Processing: {filepath}")
    raw = load_and_preprocess(filepath)

    raw_clean, epochs, epochs_clean, reject_log, ica = autoreject_and_ica(raw)

    visualize(raw_clean, epochs_clean)

    # Plot ICA overlay
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