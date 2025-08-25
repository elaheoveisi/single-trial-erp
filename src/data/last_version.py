
import os
import matplotlib.pyplot as plt
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
import pywt
from mne.time_frequency import psd_array_multitaper
import numpy as np
import openneuro
from scipy.stats import t as student_t





def get_eeg_filepath(subject_id, task_id, config):
    return f"{config['target_dir']}/sub-{subject_id}/eeg/sub-{subject_id}_task-{task_id}_eeg.bdf"




def load_eeg_data(subject_id, task_id, config=None, save_dir=None):
    if config is None:
        raise ValueError("Config must be provided.")
    # required keys (no .get)
    _ = config["target_dir"]
    _ = config["l_freq"]
    _ = config["h_freq"]
    _ = config["epoch_duration"]

    # Build path (BIDS-like)
    filepath = os.path.join(
        config["target_dir"],
        f"sub-{subject_id}",
        "eeg",
        f"sub-{subject_id}_task-{task_id}_eeg.bdf"
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BDF file not found at: {filepath}")

    # Load raw (no montage here)
    raw = mne.io.read_raw_bdf(filepath, preload=True)

    # Bandpass filter (use keys provided; no defaults here)
    raw.filter(l_freq=float(config["l_freq"]), h_freq=float(config["h_freq"]))

    # Fixed-length epochs (no montage/channel selection here)
    epochs = mne.make_fixed_length_epochs(raw, duration=float(config["epoch_duration"]), preload=True)
    print(f"Created {len(epochs)} epochs, each {epochs.tmax - epochs.tmin + 1:.1f} s long")


    return raw, epochs


def suppress_line_noise_multitaper_and_clean(subject_id, task_id, config=None):
    """
    CleanLine-style multi-taper regression for line-noise removal.
    Loads the BDF file, scans ±2 Hz around each harmonic, and removes significant sinusoids
    using regression instead of notch filtering.

    Returns a Raw object with cleaned data.
    """

    if config is None:
        raise ValueError("Config must be provided.")
    if "target_dir" not in config:
        raise ValueError("config['target_dir'] is required.")

    # --- parameters ---
    line_freq   = float(config["line_freq"])
    scan_hz     = float(config["scan_hz"])
    window_s    = float(config["window_s"])
    step_s      = float(config["step_s"])
    bandwidth   = float(config["multitaper_bandwidth"])
    p_thresh    = float(config["p_threshold"])
    max_harm    = int(config["max_harmonics"])
    picks_arg   = config["picks"]
    sel_chs     = config["selected_channels"]

    # --- build path & load ---
    filepath = f"{config['target_dir']}/sub-{subject_id}/eeg/sub-{subject_id}_task-{task_id}_eeg.bdf"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BDF file not found at: {filepath}")

    raw = mne.io.read_raw_bdf(filepath, preload=True)
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="ignore")

    # optional channel selection
    if sel_chs is not None:
        keep = [ch for ch in raw.ch_names if ch in sel_chs and ch in montage.ch_names]
        if not keep:
            raise ValueError("No selected_channels found in data.")
        raw.pick_channels(keep)
    else:
        raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])

    # resolve picks to indices
    if isinstance(picks_arg, str) and picks_arg.lower() == "eeg":
        ch_idxs = mne.pick_types(raw.info, eeg=True, exclude=[])
    elif isinstance(picks_arg, (list, tuple, np.ndarray)):
        ch_idxs = mne.pick_channels(raw.ch_names, include=list(picks_arg), exclude=[])
    else:
        ch_idxs = np.arange(len(raw.ch_names))

    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=ch_idxs)
    n_ch, n_samp = data.shape

    win_len = int(round(window_s * sfreq))
    hop = int(round(step_s * sfreq))
    if win_len <= 0 or win_len > n_samp:
        raise ValueError("Invalid window length.")

    t = (np.arange(win_len) - win_len / 2) / sfreq
    starts = np.arange(0, n_samp - win_len + 1, hop)

    print(f"[CleanLine-like] {len(starts)} windows, win/hop={window_s}/{step_s}s, "
          f"scan=±{scan_hz} Hz, f0={line_freq} Hz, harmonics={max_harm}")

    cleaned = data.copy()

    for ch_i in range(n_ch):
        x = data[ch_i]
        for s0 in starts:
            seg = x[s0:s0 + win_len]
            # harmonics loop
            for h in range(1, max_harm + 1):
                target = line_freq * h
                if target >= sfreq / 2:
                    continue

                # estimate peak frequency around target
                fmin = max(target - scan_hz, 0.1)
                fmax = target + scan_hz
                psd, freqs = psd_array_multitaper(
                    seg[np.newaxis, :], sfreq=sfreq,
                    fmin=fmin, fmax=fmax,
                    bandwidth=bandwidth, adaptive=False,
                    normalization="full", verbose=False
                )
                f_peak = float(freqs[int(np.argmax(psd[0]))])

                # build design matrix for regression
                s = np.sin(2 * np.pi * f_peak * t)
                c = np.cos(2 * np.pi * f_peak * t)
                X = np.vstack([s, c]).T
                beta, _, _, _ = np.linalg.lstsq(X, seg, rcond=None)
                y_hat = X @ beta
                resid = seg - y_hat

                # t-test for coefficients
                dof = max(len(seg) - 2, 1)
                sigma2 = float(np.dot(resid, resid)) / dof
                XtX_inv = np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(XtX_inv) * sigma2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_vals = beta / se
                t_max = np.nanmax(np.abs(t_vals))
                p_val = 2.0 * student_t.sf(t_max, dof)

                # remove if significant
                if p_val <= p_thresh:
                    cleaned[ch_i, s0:s0 + win_len] = resid
                    seg = resid  # update for higher harmonics

        if n_ch >= 10 and (ch_i + 1) % max(1, n_ch // 10) == 0:
            print(f"  processed channel {ch_i+1}/{n_ch}")

    raw_clean = raw.copy()
    raw_clean._data[ch_idxs, :] = cleaned
    return raw_clean




def wavelet_denoise_epochs_from_epochs(subject_id, task_id, config=None, save_dir=None):
    """
    Wavelet-enhanced ICA (W-ICA) denoising using Picard ICA.

    Steps:
      1) Run ICA (Picard) to decompose EEG into components
      2) Apply SWT (coif5, level=5) to each component
         - Compute one global soft threshold per component:
           Thr = median(|D|) / 0.6745 * sqrt(2*log(N))
         - Apply threshold only to detail coefficients
      3) Inverse SWT to reconstruct denoised component series
      4) Inverse ICA to get cleaned epochs
    """
    if config is None:
        raise ValueError("Config must be provided.")

    wavelet = "coif5"
    level = 5

    # === Load epochs ===
    input_path = os.path.join(
        config["processed_dir"], f"sub-{subject_id}_task-{task_id}_epo.fif"
    )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Epochs file not found at: {input_path}")

    epochs = mne.read_epochs(input_path, preload=True)

    # === ICA (Picard) ===
    ica = ICA(
        n_components=min(20, len(epochs.ch_names)),
        method="picard",
        random_state=42,
        fit_params=dict(max_iter="auto")
    )
    ica.fit(epochs)

    # Shape: (n_epochs, n_components, n_times)
    sources = ica.get_sources(epochs).get_data()
    n_ep, n_comp, _ = sources.shape
    denoised_sources = np.empty_like(sources)

    # === Wavelet denoising per component ===
    for ep in range(n_ep):
        for c in range(n_comp):
            signal = sources[ep, c, :]

            # Determine SWT level
            used_level = min(level, pywt.swt_max_level(len(signal)))
            coeffs = pywt.swt(signal, wavelet, level=used_level)

            # Combine all detail coeffs for global threshold
            all_D = np.concatenate([cD for (_, cD) in coeffs]) if coeffs else np.array([])
            N = len(signal)
            sigma_hat = np.median(np.abs(all_D)) / 0.6745 if all_D.size else 0.0
            Thr = sigma_hat * np.sqrt(2.0 * np.log(max(N, 2)))

            # Apply soft threshold to detail coeffs only
            thr_coeffs = []
            for (cA, cD) in coeffs:
                cD_thr = pywt.threshold(cD, Thr, mode="soft")
                thr_coeffs.append((cA, cD_thr))

            # Inverse SWT
            denoised_sources[ep, c, :] = pywt.iswt(thr_coeffs, wavelet)

    # === Inverse ICA to reconstruct cleaned EEG ===
    cleaned_epochs = ica.inverse_transform(
        mne.EpochsArray(denoised_sources, info=epochs.info.copy(), tmin=epochs.tmin)
    )

    # === Save if requested ===
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(
            save_dir, f"sub-{subject_id}_task-{task_id}_epo_wica_picard.fif"
        )
        cleaned_epochs.save(out_path, overwrite=True)

    print(f"W-ICA (Picard) denoising complete for subject {subject_id}, task {task_id}")
    return cleaned_epochs



def clean_eeg(subject_id, task_id, config=None, save_dir=None):
    if config is None:
        raise ValueError("Config must be provided.")

    save_plot_dir = config.get("save_plot_dir", "outputs")
    os.makedirs(save_plot_dir, exist_ok=True)
    filepath = get_eeg_filepath(subject_id, task_id, config)
    print(f"Processing: {filepath}")

    # 1) Get line-noise–suppressed RAW using your existing signature
    raw_clean = suppress_line_noise_multitaper_and_clean(subject_id, task_id, config)

    # 2) Build epochs from the CLEAN raw (so epochs match the cleaned signal)
    epochs = mne.make_fixed_length_epochs(
        raw_clean, duration=float(config["epoch_duration"]), preload=True
    )
    print(f"Created {len(epochs)} epochs, each {epochs.tmax - epochs.tmin + 1:.1f} s long")

    # 3) AutoReject
    ar = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=True)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)
    if len(reject_log.bad_epochs) > 0:
        print(f"{len(reject_log.bad_epochs)} bad epochs found.")

    # 4) Bad-epoch plot
    fig_bad = epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6), show=False)
    fig_bad.suptitle(f"Bad Epochs - Subject {subject_id} Task {task_id}")
    fig_path = os.path.join(save_plot_dir, f"sub-{subject_id}_task-{task_id}_bad_epochs.png")
    try:
        fig_bad.savefig(fig_path)
        print(f"Saved bad epoch plot to {fig_path}")
    except PermissionError:
        print(f"Permission denied: Cannot save plot to '{fig_path}'.")
    plt.close(fig_bad)
    reject_log.plot("horizontal"); plt.close()

    # 5) ICA on cleaned epochs, apply to both raw_clean and epochs_clean
    ica = ICA(n_components=20, method="picard", random_state=0)
    ica.fit(epochs_clean)

    ica.plot_components(inst=epochs_clean, show=False); plt.close()
    ica.plot_overlay(epochs_clean.average(), exclude=ica.exclude, show=False); plt.close()

    raw_clean = ica.apply(raw_clean.copy())
    epochs_clean = ica.apply(epochs_clean.copy())

    # 6) Evoked & save
    evoked = epochs_clean.average()
    print("Evoked shape:", evoked.data.shape)
    evoked._data *= 1e6
    fig_evoked = evoked.plot(scalings=dict(eeg=20), time_unit="s", show=False)
    fig_evoked.suptitle(f"Evoked - Subject {subject_id} Task {task_id}")
    fig_evoked.savefig(os.path.join(save_plot_dir, f"sub-{subject_id}_task-{task_id}_evoked.png"))
    plt.close(fig_evoked)

    if len(reject_log.bad_epochs) > 0:
        evoked_bad = epochs[reject_log.bad_epochs].average()
        plt.figure()
        plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, "r", zorder=-1)
        evoked_clean = epochs_clean.average()
        evoked_clean.plot(axes=plt.gca(), show=False)
        plt.title(f"Bad vs Clean - Subject {subject_id} Task {task_id}")
        plt.savefig(os.path.join(save_plot_dir, f"sub-{subject_id}_task-{task_id}_bad_vs_clean.png"))
        plt.close()

    return raw_clean, epochs_clean


# Preprocess All Subjects & Tasks 
def preprocess_all_subjects_tasks(subjects, tasks, config=None, save_dir=None):
    for subject_id in subjects:
        for task_id in tasks:
            raw_clean, epochs_clean = clean_eeg(
                subject_id, task_id, config=config
            )
            path = f"{config['processed_dir']}/sub-{subject_id}_task-{task_id}_cleaned_raw.fif"
            raw_clean.save(path, overwrite=True)
            print(f"Saved cleaned data to: {path}")
