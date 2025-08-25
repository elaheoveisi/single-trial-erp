import os
import matplotlib.pyplot as plt
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
import pywt
from mne.time_frequency import psd_array_multitaper
import numpy as np
from scipy.stats import t as student_t
# FIX: correct import
from mne_icalabel import label_components



def get_eeg_filepath(subject_id, task_id, config):
    return f"{config['target_dir']}/sub-{subject_id}/eeg/sub-{subject_id}_task-{task_id}_eeg.bdf"


def load_eeg_data(subject_id, task_id, config=None, save_dir=None):
    if config is None:
        raise ValueError("Config must be provided.")
    
    _ = config["target_dir"]
    _ = config["l_freq"]
    _ = config["h_freq"]
    _ = config["epoch_duration"]

    # ALWAYS use the path helper
    filepath = get_eeg_filepath(subject_id, task_id, config)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BDF file not found at: {filepath}")

    # RAW cache: read once, reuse elsewhere
    raw = config.get("_raw_cache")
    if raw is None or raw.filenames[0] != filepath:
        raw = mne.io.read_raw_bdf(filepath, preload=True)
        montage = mne.channels.make_standard_montage("biosemi64")
        raw.set_montage(montage, on_missing="ignore")

        # optional channel selection once
        sel_chs = config.get("selected_channels")
        if sel_chs is not None:
            keep = [ch for ch in raw.ch_names if ch in sel_chs and ch in montage.ch_names]
            if not keep:
                raise ValueError("No selected_channels found in data.")
            raw.pick_channels(keep)
        else:
            raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])

        raw.filter(l_freq=float(config["l_freq"]), h_freq=float(config["h_freq"]))
        config["_raw_cache"] = raw  # cache for reuse

    # EPOCHS cache
    epochs = config.get("_epochs_cache")
    epoch_key = (filepath, float(config["epoch_duration"]))
    if epochs is None or config.get("_epochs_key") != epoch_key:
        epochs = mne.make_fixed_length_epochs(raw, duration=float(config["epoch_duration"]), preload=True)
        print(f"Created {len(epochs)} epochs, each {epochs.tmax - epochs.tmin + 1:.1f} s long")
        config["_epochs_cache"] = epochs
        config["_epochs_key"] = epoch_key

    return raw, epochs


def suppress_line_noise_multitaper_and_clean(subject_id, task_id, config=None):
    if config is None:
        raise ValueError("Config must be provided.")
    if "target_dir" not in config:
        raise ValueError("config['target_dir'] is required.")

    line_freq   = float(config["line_freq"])
    scan_hz     = float(config["scan_hz"])
    window_s    = float(config["window_s"])
    step_s      = float(config["step_s"])
    bandwidth   = float(config["multitaper_bandwidth"])
    p_thresh    = float(config["p_threshold"])
    max_harm    = int(config["max_harmonics"])
    picks_arg   = config["picks"]

    #  Read RAW once via cache from load_eeg_data (centralized montage, picks, bandpass)
    filepath = get_eeg_filepath(subject_id, task_id, config)
    raw = config.get("_raw_cache")
    if raw is None or raw.filenames[0] != filepath:
        # if not loaded yet, load and cache quickly (mirror load_eeg_data behavior minimally)
        raw = mne.io.read_raw_bdf(filepath, preload=True)
        montage = mne.channels.make_standard_montage("biosemi64")
        raw.set_montage(montage, on_missing="ignore")
        sel_chs = config.get("selected_channels")
        if sel_chs is not None:
            keep = [ch for ch in raw.ch_names if ch in sel_chs and ch in montage.ch_names]
            if not keep:
                raise ValueError("No selected_channels found in data.")
            raw.pick_channels(keep)
        else:
            raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])
        # if bandpass is expected here too
        if "l_freq" in config and "h_freq" in config:
            raw.filter(l_freq=float(config["l_freq"]), h_freq=float(config["h_freq"]))
        config["_raw_cache"] = raw

    # ---- Resolve picks once, cache them
    picks_cache_key = ("_picks_cache", tuple(raw.ch_names), str(picks_arg))
    if config.get("_picks_cache_key") != picks_cache_key:
        if isinstance(picks_arg, str) and picks_arg.lower() == "eeg":
            ch_idxs = mne.pick_types(raw.info, eeg=True, exclude=[])
        elif isinstance(picks_arg, (list, tuple, np.ndarray)):
            ch_idxs = mne.pick_channels(raw.ch_names, include=list(picks_arg), exclude=[])
        else:
            ch_idxs = np.arange(len(raw.ch_names))
        config["_picks_cache"] = ch_idxs
        config["_picks_cache_key"] = picks_cache_key
    else:
        ch_idxs = config["_picks_cache"]

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
            for h in range(1, max_harm + 1):
                target = line_freq * h
                if target >= sfreq / 2:
                    continue

                fmin = max(target - scan_hz, 0.1)
                fmax = target + scan_hz
                psd, freqs = psd_array_multitaper(
                    seg[np.newaxis, :], sfreq=sfreq,
                    fmin=fmin, fmax=fmax,
                    bandwidth=bandwidth, adaptive=False,
                    normalization="full", verbose=False
                )
                f_peak = float(freqs[int(np.argmax(psd[0]))])

                s = np.sin(2 * np.pi * f_peak * t)
                c = np.cos(2 * np.pi * f_peak * t)
                X = np.vstack([s, c]).T
                beta, _, _, _ = np.linalg.lstsq(X, seg, rcond=None)
                y_hat = X @ beta
                resid = seg - y_hat

                dof = max(len(seg) - 2, 1)
                sigma2 = float(np.dot(resid, resid)) / dof
                XtX_inv = np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(XtX_inv) * sigma2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_vals = beta / se
                t_max = np.nanmax(np.abs(t_vals))
                p_val = 2.0 * student_t.sf(t_max, dof)

                if p_val <= p_thresh:
                    cleaned[ch_i, s0:s0 + win_len] = resid
                    seg = resid  # update for higher harmonics

        if n_ch >= 10 and (ch_i + 1) % max(1, n_ch // 10) == 0:
            print(f"  processed channel {ch_i+1}/{n_ch}")

    raw_clean = raw.copy()
    raw_clean._data[ch_idxs, :] = cleaned
    return raw_clean


def wavelet_denoise_epochs_from_epochs(subject_id, task_id, config=None):
    if config is None:
        raise ValueError("Config must be provided.")
    for k in ["target_dir", "epoch_duration"]:
        if k not in config:
            raise ValueError(f"config['{k}'] is required.")

    # ---- Reuse RAW and EPOCHS from cache if present; otherwise load once
    filepath = get_eeg_filepath(subject_id, task_id, config)
    raw = config.get("_raw_cache")
    if raw is None or raw.filenames[0] != filepath:
        raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
        montage = mne.channels.make_standard_montage("biosemi64")
        raw.set_montage(montage, on_missing="ignore")
        sel_chs = config.get("selected_channels")
        if sel_chs is not None:
            keep = [ch for ch in raw.ch_names if ch in sel_chs and ch in montage.ch_names]
            if not keep:
                raise ValueError("No selected_channels found in data.")
            raw.pick_channels(keep)
        else:
            raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])
        config["_raw_cache"] = raw

    epochs = config.get("_epochs_cache")
    epoch_key = (filepath, float(config["epoch_duration"]))
    if epochs is None or config.get("_epochs_key") != epoch_key:
        epoch_dur = float(config["epoch_duration"])
        epochs = mne.make_fixed_length_epochs(
            raw, duration=epoch_dur, preload=True, verbose=False
        )
        print(f"Created {len(epochs)} epochs, each {epochs.tmax - epochs.tmin + 1:.1f} s long")
        config["_epochs_cache"] = epochs
        config["_epochs_key"] = epoch_key

    # ---- ICA (Picard)
    n_comp = min(20, len(epochs.ch_names))
    ica = ICA(n_components=n_comp, method="picard", random_state=42, max_iter=500, verbose=False)
    ica.fit(epochs)

    # Sources: (n_epochs, n_comp, n_times)
    src = ica.get_sources(epochs).get_data()
    n_ep, n_comp, _ = src.shape

    # ---- Wavelet (SWT) denoising on sources
    wavelet = "coif5"
    max_level = 5
    den = np.empty_like(src)

    for e in range(n_ep):
        for c in range(n_comp):
            sig = src[e, c, :]
            level = min(max_level, pywt.swt_max_level(len(sig)))
            if level < 1:
                den[e, c, :] = sig
                continue

            coeffs = pywt.swt(sig, wavelet, level=level, trim_approx=False)

            # Normalize coeffs to (cA, cD) pairs robustly
            pairs = []
            for elem in coeffs:
                if isinstance(elem, (list, tuple)) and len(elem) == 2:
                    cA, cD = elem
                else:
                    cA, cD = elem[0], elem[1]
                pairs.append((cA, cD))

            # Estimate sigma from all detail coeffs
            if pairs:
                detail_list = [cD for (_, cD) in pairs]
                all_D = np.concatenate([d.ravel() for d in detail_list]) if detail_list else np.array([])
                sigma = (np.median(np.abs(all_D)) / 0.6745) if all_D.size else 0.0
            else:
                sigma = 0.0
            Thr = sigma * np.sqrt(2.0 * np.log(max(len(sig), 2)))

            # Soft-threshold detail coeffs, keep approximations
            thr_pairs = [(cA, pywt.threshold(cD, Thr, mode="soft")) for (cA, cD) in pairs]
            rec = pywt.iswt(thr_pairs, wavelet)
            if rec.shape[-1] != sig.shape[-1]:
                rec = rec[..., : sig.shape[-1]]
            den[e, c, :] = rec

    # ---- Reconstruct sensor-space epochs directly via the mixing matrix
    # A: (n_channels, n_comp)
    A = ica.mixing_matrix_

    # Ensure orientation is (n_channels, n_comp)
    if A.shape[0] != len(epochs.ch_names) and A.shape[1] == len(epochs.ch_names):
        A = A.T

    if A.shape[0] != len(epochs.ch_names):
        raise ValueError(
            f"Mixing matrix channels ({A.shape[0]}) != epochs channels ({len(epochs.ch_names)})."
        )
    if A.shape[1] != den.shape[1]:
        raise ValueError(
            f"Mixing matrix component count ({A.shape[1]}) != denoised source count ({den.shape[1]})."
        )

    # den: (n_epochs, n_comp, n_times) -> X_rec: (n_epochs, n_channels, n_times)
    X_rec = np.einsum("fc,ect->eft", A, den)

    cleaned_epochs = mne.EpochsArray(
        X_rec, info=epochs.info.copy(), tmin=epochs.tmin, verbose=False
    )

    if config.get("processed_dir"):
        os.makedirs(config["processed_dir"], exist_ok=True)
        outp = os.path.join(
            config["processed_dir"],
            f"sub-{subject_id}_task-{task_id}_epo_wica_picard.fif",
        )
        cleaned_epochs.save(outp, overwrite=True)
        print(f"Saved W-ICA epochs to: {outp}")

    return cleaned_epochs


def clean_eeg(subject_id, task_id, config=None, save_dir=None):

    try:
        from mne_icalabel import label_components
    except Exception as e:
        raise RuntimeError(
            "mne_icalabel is required for ICLabel. Install with `pip install mne-icalabel`."
        ) from e

    if config is None:
        raise ValueError("Config must be provided.")

    save_plot_dir = config.get("save_plot_dir", "outputs")
    os.makedirs(save_plot_dir, exist_ok=True)
    filepath = get_eeg_filepath(subject_id, task_id, config)
    print(f"Processing: {filepath}")

    # READ once & cache (reused by other functions)
    raw = config.get("_raw_cache")
    if raw is None or raw.filenames[0] != filepath:
        raw = mne.io.read_raw_bdf(filepath, preload=True)
        montage = mne.channels.make_standard_montage("biosemi64")
        raw.set_montage(montage, on_missing="ignore")
        sel_chs = config.get("selected_channels")
        if sel_chs is not None:
            keep = [ch for ch in raw.ch_names if ch in sel_chs and ch in montage.ch_names]
            if not keep:
                raise ValueError("No selected_channels found in data.")
            raw.pick_channels(keep)
        else:
            raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])
        if "l_freq" in config and "h_freq" in config:
            raw.filter(l_freq=float(config["l_freq"]), h_freq=float(config["h_freq"]))
        config["_raw_cache"] = raw

    # 1) Line-noise suppression (operates in-memory)
    raw_clean = suppress_line_noise_multitaper_and_clean(subject_id, task_id, config)

    # 2) Epochs (centralized via a small cache)
    epochs = config.get("_epochs_cache")
    epoch_key = (filepath, float(config["epoch_duration"]))
    if epochs is None or config.get("_epochs_key") != epoch_key:
        epochs = mne.make_fixed_length_epochs(
            raw_clean, duration=float(config["epoch_duration"]), preload=True
        )
        print(f"Created {len(epochs)} epochs, each {epochs.tmax - epochs.tmin:.1f} s long")
        config["_epochs_cache"] = epochs
        config["_epochs_key"] = epoch_key
    else:
        if epochs._raw is not raw_clean:  # remake if cached from pre-clean stage
            epochs = mne.make_fixed_length_epochs(
                raw_clean, duration=float(config["epoch_duration"]), preload=True
            )
            print(f"Created {len(epochs)} epochs, each {epochs.tmax - epochs.tmin:.1f} s long")
            config["_epochs_cache"] = epochs
            config["_epochs_key"] = epoch_key

    # 3) AutoReject
    ar = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=True)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)
    n_bad = len(reject_log.bad_epochs) if hasattr(reject_log, "bad_epochs") else 0
    if n_bad > 0:
        print(f"{n_bad} bad epochs found.")

    # 4) Bad-epoch plot
    dpi = int(config.get("dpi", 200))
    if n_bad > 0:
        try:
            fig_bad = epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6), show=False)
            fig_bad.suptitle(f"Bad Epochs - Subject {subject_id} Task {task_id}")
            fig_path = os.path.join(save_plot_dir, f"sub-{subject_id}_task-{task_id}_bad_epochs.png")
            fig_bad.savefig(fig_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved bad epoch plot to {fig_path}")
            plt.close(fig_bad)
        except Exception as e:
            print(f"Could not plot bad epochs: {e}")
    try:
        reject_log.plot("horizontal")
        plt.close()
    except Exception:
        pass

    # 5) ICA (PICARD), with ICLabel preconditions (CAR + 1–100 Hz)
    epochs_for_ica = epochs_clean.copy()
    epochs_for_ica.set_eeg_reference("average")
    epochs_for_ica.filter(l_freq=1.0, h_freq=100.0, fir_design="firwin", phase="zero")

    n_comp_cfg = int(config.get("ica_n_components", 20))
    n_ch = len(epochs_for_ica.ch_names)
    n_comp = min(n_comp_cfg, n_ch)  # clamp

    ica = ICA(
        n_components=n_comp,
        method="picard",
        random_state=int(config.get("random_state", 97)),
        fit_params=dict(extended=True, ortho=False),
        max_iter="auto",
    )
    ica.fit(epochs_for_ica)

    # ICLabel classification
    try:
        ica_labels, ica_probs = label_components(epochs_for_ica, ica, method="iclabel")
        labels = np.atleast_1d(np.asarray(ica_labels)).ravel()
        probs = np.atleast_2d(np.asarray(ica_probs, dtype=float))
    except Exception as e:
        print(f"ICLabel labeling failed, but continuing with a fallback: {e}")
        labels = np.array([])
        probs = np.array([])

    class_names = np.array(
        ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]
    )
    pred_idx = probs.argmax(axis=1) if probs.size > 0 else np.array([], dtype=int)

    if pred_idx.size == labels.size and (pred_idx.size == 0 or pred_idx.max() < class_names.size):
        pred_names = class_names[pred_idx]
        pred_probs = probs[np.arange(pred_idx.size), pred_idx] if probs.size > 0 else np.zeros(pred_idx.size, dtype=float)
    else:
        pred_names = labels
        pred_probs = np.ones(labels.shape[0], dtype=float)

    pred_probs = np.asarray(pred_probs, dtype=float)  # final guard

    bad_classes = set(config.get("iclabel_bad_classes", ["eye", "muscle", "heart"]))
    prob_thresh = float(config.get("iclabel_prob_threshold", 0.0))
    bad_mask = np.isin(pred_names, list(bad_classes)) & (pred_probs >= prob_thresh)
    bad_idx = np.flatnonzero(bad_mask)

    ica.exclude = bad_idx.tolist()
    print(f"ICLabel predicted bad components: {ica.exclude} "
          f"({pred_names[bad_mask].tolist() if bad_idx.size else []})")

    # Apply ICA
    raw_clean = ica.apply(raw_clean.copy())
    epochs_clean = ica.apply(epochs_clean.copy())

    # 6) Evoked & save
    evoked = epochs_clean.average()
    print("Evoked shape:", evoked.data.shape)
    evoked_to_plot = evoked.copy()
    evoked_to_plot.data *= 1e6
    fig_evoked = evoked_to_plot.plot(scalings=dict(eeg=20), time_unit="s", show=False)
    fig_evoked.suptitle(f"Evoked - Subject {subject_id} Task {task_id}")
    fig_ev_path = os.path.join(save_plot_dir, f"sub-{subject_id}_task-{task_id}_evoked.png")
    try:
        fig_evoked.savefig(fig_ev_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved evoked plot to {fig_ev_path}")
    finally:
        plt.close(fig_evoked)

    if n_bad > 0:
        try:
            evoked_bad = epochs[reject_log.bad_epochs].average()
            plt.figure()
            plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, "r", zorder=-1)
            evoked_clean_avg = epochs_clean.average()
            evoked_clean_avg.plot(axes=plt.gca(), show=False)
            plt.title(f"Bad vs Clean - Subject {subject_id} Task {task_id}")
            bad_clean_path = os.path.join(
                save_plot_dir, f"sub-{subject_id}_task-{task_id}_bad_vs_clean.png"
            )
            plt.gcf().savefig(bad_clean_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            print(f"Saved bad vs clean plot to {bad_clean_path}")
        except Exception as e:
            print(f"Could not create bad vs clean plot: {e}")

    return raw_clean, epochs_clean




# Preprocess All Subjects & Tasks 
def preprocess_all_subjects_tasks(subjects, tasks, config=None, save_dir=None):
    for subject_id in subjects:
        for task_id in tasks:
            raw_clean, epochs_clean = clean_eeg(
                subject_id, task_id, config=config
            )
            path = f"{config['processed_dir']}/sub-{subject_id}_task-{task_id}_cleaned_raw.fif"
            os.makedirs(config["processed_dir"], exist_ok=True)
            raw_clean.save(path, overwrite=True)
            print(f"Saved cleaned data to: {path}")
