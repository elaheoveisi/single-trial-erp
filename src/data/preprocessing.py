import os

import matplotlib.pyplot as plt
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA


#  EEG Load + Preprocess
def load_eeg_data(filepath):
    raw = mne.io.read_raw_bdf(filepath, preload=True)
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="ignore")
    raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])
    raw.filter(l_freq=1.0, h_freq=60.0)
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    return raw, epochs


#  Clean EEG
def clean_eeg(subject_id, task_id, config=None):
    filepath = f"{config['target_dir']}/sub-{subject_id}/eeg/sub-{subject_id}_task-{task_id}_eeg.bdf"
    if not filepath:
        raise ValueError(f"No file found for subject {subject_id}, task {task_id}")
    print(f"Processing: {filepath}")

    raw, epochs = load_eeg_data(filepath)

    ar = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=True)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)

    if len(reject_log.bad_epochs) > 0:
        print(f"{len(reject_log.bad_epochs)} bad epochs found.")
        fig_bad = epochs[reject_log.bad_epochs].plot(
            scalings=dict(eeg=100e-6), show=False
        )
        fig_bad.suptitle(f"Bad Epochs - Subject {subject_id} Task {task_id}")
        if config["save_plot_dir"]:
            fig_bad.savefig(
                os.path.join(
                    config["save_plot_dir"],
                    f"/sub-{subject_id}_task-{task_id}_bad_epochs.png",
                )
            )
        plt.show()
        reject_log.plot("horizontal")

    ica = ICA(n_components=64, method="picard", random_state=0)
    ica.fit(epochs_clean)
    ica.plot_components(inst=raw)
    plt.suptitle(f"ICA Components - Subject {subject_id} Task {task_id}")
    plt.show()

    ica.plot_overlay(epochs_clean.average(), exclude=ica.exclude)
    plt.suptitle(f"ICA Overlay - Subject {subject_id} Task {task_id}")
    plt.show()

    ica.exclude = []  # Optional manual edit
    raw_clean = ica.apply(raw.copy())
    ica.apply(epochs_clean)

    raw_clean.plot(
        scalings="auto", title=f"Cleaned Raw - Subject {subject_id} Task {task_id}"
    )
    plt.show()

    evoked = epochs_clean.average()
    print("Evoked shape:", evoked.data.shape)
    evoked._data *= 1e6
    fig_evoked = evoked.plot(scalings=dict(eeg=20), time_unit="s", show=False)
    fig_evoked.suptitle(f"Evoked - Subject {subject_id} Task {task_id}")
    if config["save_plot_dir"]:
        fig_evoked.savefig(
            os.path.join(
                config["save_plot_dir"], f"sub-{subject_id}_task-{task_id}_evoked.png"
            )
        )
    plt.show()

    if len(reject_log.bad_epochs) > 0:
        evoked_bad = epochs[reject_log.bad_epochs].average()
        plt.figure()
        plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, "r", zorder=-1)
        evoked_clean = epochs_clean.average()
        evoked_clean.plot(axes=plt.gca(), show=False)
        plt.title(f"Bad vs Clean - Subject {subject_id} Task {task_id}")
        if config["save_plot_dir"]:
            plt.savefig(
                os.path.join(
                    config["save_plot_dir"],
                    f"sub-{subject_id}_task-{task_id}_bad_vs_clean.png",
                )
            )
        plt.show()

    return raw_clean, epochs_clean


# Preprocess all the subjects and tasks
def preprocess_all_subjects_tasks(subjects, tasks, config=None):
    for subject_id in subjects:
        for task_id in tasks:
            raw_clean, epochs_clean = clean_eeg(
                subject_id, task_id, save_plot_dir=config["save_plot_dir"]
            )
            path = f"{config['processed_dir']}/sub-{subject_id}_task-{task_id}_cleaned_raw.fif"
            raw_clean.save(path, overwrite=True)
            print(f"Saved cleaned data to: {path}")
