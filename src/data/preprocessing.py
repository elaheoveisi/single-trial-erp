import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
import pywt


# ----------------------------- Paths & Loading -----------------------------
def get_eeg_filepath(subject_id, task_id, config):
    return f"{config['target_dir']}/sub-{subject_id}/eeg/sub-{subject_id}_task-{task_id}_eeg.bdf"


def load_eeg_data(subject_id, task_id, config=None, save_dir=None):
    if config is None:
        raise ValueError("Config must be provided.")
    _ = config["target_dir"]; _ = config["l_freq"]; _ = config["h_freq"]; _ = config["epoch_duration"]

    filepath = get_eeg_filepath(subject_id, task_id, config)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BDF file not found at: {filepath}")

    # RAW cache
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
            raw.pick(keep)  # modern API
        else:
            raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])

        raw.filter(l_freq=float(config["l_freq"]), h_freq=float(config["h_freq"]))
        config["_raw_cache"] = raw

    # EPOCHS cache
    epochs = config.get("_epochs_cache")
    epoch_key = (filepath, float(config["epoch_duration"]))
    if epochs is None or config.get("_epochs_key") != epoch_key:
        epochs = mne.make_fixed_length_epochs(raw, duration=float(config["epoch_duration"]), preload=True)
        print(f"Created {len(epochs)} epochs, each {epochs.tmax - epochs.tmin:.1f} s long")
        config["_epochs_cache"] = epochs
        config["_epochs_key"] = epoch_key

    return raw, epochs


# --------- CleanLine-style multitaper suppression (±2 Hz, 4 s, 1 s, τ=100, p=0.01) ---------
def suppress_line_noise_multitaper_and_clean(subject_id, task_id, config=None):
    if config is None:
        raise ValueError("Config must be provided.")
    from scipy.signal.windows import dpss
    from scipy.stats import f as fdist

    filepath = get_eeg_filepath(subject_id, task_id, config)
    raw = config.get("_raw_cache")
    if raw is None or raw.filenames[0] != filepath:
        raw = mne.io.read_raw_bdf(filepath, preload=True)
        montage = mne.channels.make_standard_montage("biosemi64")
        raw.set_montage(montage, on_missing="ignore")
        sel = config.get("selected_channels")
        if sel is not None:
            keep = [ch for ch in raw.ch_names if ch in sel and ch in montage.ch_names]
            if not keep:
                raise ValueError("No selected_channels found in data.")
            raw.pick(keep)
        else:
            raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])
        if "l_freq" in config and "h_freq" in config:
            raw.filter(l_freq=float(config["l_freq"]), h_freq=float(config["h_freq"]))
        config["_raw_cache"] = raw

    fs   = raw.info["sfreq"]
    f0   = float(config.get("line_freq", 60.0))
    win  = float(config.get("clean_win_s", 4.0))
    hop  = float(config.get("clean_hop_s", 1.0))
    scan = float(config.get("clean_scan_hz", 2.0))
    tau  = float(config.get("clean_tau", 100.0))
    pth  = float(config.get("clean_p", 0.01))
    NW   = float(config.get("clean_NW", 3.5))

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if picks.size == 0:
        return raw

    X = raw.get_data(picks=picks, reject_by_annotation="omit")
    n_ch, n = X.shape
    wlen, hop_samp = int(win*fs), int(hop*fs)
    if wlen <= 8 or hop_samp <= 0:
        return raw

    K = max(1, int(2*NW - 1))
    tap = dpss(wlen, NW, K)
    nfft = 1 << int(np.ceil(np.log2(wlen*4)))
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    band = (freqs >= max(0, f0 - scan)) & (freqs <= f0 + scan)
    if not np.any(band):
        return raw

    alpha = np.exp(-1.0 / max(1.0, tau))
    corr = np.zeros_like(X); W = np.zeros((n_ch, n))
    win_hann = np.hanning(wlen)

    for s in range(0, n - wlen + 1, hop_samp):
        idx = slice(s, s + wlen)
        seg = X[:, idx]
        mean_seg = seg.mean(axis=0)

        P = 0.0
        for v in tap:
            Xf = np.fft.rfft(mean_seg * v, nfft); P += (np.abs(Xf)**2)
        P /= K
        Ps = P.copy()
        for i in range(1, Ps.size):
            Ps[i] = alpha*Ps[i-1] + (1 - alpha)*Ps[i]

        fk = freqs[band][np.argmax(Ps[band])]
        t = np.arange(wlen)/fs
        C = np.column_stack((np.cos(2*np.pi*fk*t), np.sin(2*np.pi*fk*t)))
        pinv = np.linalg.pinv(C)
        beta = X[:, idx] @ pinv.T
        yhat = beta @ C.T

        rss1 = np.sum((seg - yhat)**2, axis=1); rss0 = np.sum(seg**2, axis=1)
        df1, df2 = 2, max(1, wlen - 2)
        F = ((rss0 - rss1)/df1) / (rss1/df2)
        pvals = 1 - fdist.cdf(F, df1, df2)

        m = pvals < pth
        if np.any(m):
            corr[m, s:s+wlen] += yhat[m] * win_hann
            W[m, s:s+wlen]    += win_hann

    W[W == 0] = 1.0
    X_clean = X - corr / W
    raw._data[picks] = X_clean
    return raw


# -------------------------- Wavelet denoising on ICs --------------------------
def wavelet_denoise_epochs_from_epochs(subject_id, task_id, config=None):
    if config is None:
        raise ValueError("Config must be provided.")
    raw, epochs = load_eeg_data(subject_id, task_id, config)

    n_comp = min(20, len(epochs.ch_names))
    ica = ICA(n_components=n_comp, method="picard", random_state=42, max_iter=500, verbose=False)
    ica.fit(epochs)

    src = ica.get_sources(epochs).get_data()  # (n_epochs, n_comp, n_times)
    n_ep, n_comp, n_times = src.shape

    wavelet = "coif5"; max_level = 5
    den = np.empty_like(src)
    for e in range(n_ep):
        for c in range(n_comp):
            sig = src[e, c, :]
            level = min(max_level, pywt.swt_max_level(len(sig)))
            if level < 1:
                den[e, c, :] = sig; continue
            coeffs = pywt.swt(sig, wavelet, level=level, trim_approx=False)
            detail_coeffs = [cD for cA, cD in coeffs]
            all_D = np.concatenate([d.ravel() for d in detail_coeffs]) if detail_coeffs else np.array([])
            sigma = (np.median(np.abs(all_D)) / 0.6745) if all_D.size else 0.0
            thr = sigma * np.sqrt(2.0 * np.log(max(len(sig), 2)))
            thr_coeffs = [(cA, pywt.threshold(cD, thr, mode="soft")) for cA, cD in coeffs]
            rec = pywt.iswt(thr_coeffs, wavelet)
            den[e, c, :] = rec[:n_times]

    cleaned_epochs = epochs.copy()
    cleaned_epochs._data = np.einsum("fc,ect->eft", ica.mixing_matrix_, den)

    if config.get("processed_dir"):
        os.makedirs(config["processed_dir"], exist_ok=True)
        outp = os.path.join(config["processed_dir"],
                            f"sub-{subject_id}_task-{task_id}_desc-wica-picard_epo.fif")
        cleaned_epochs.save(outp, overwrite=True)
        print(f"Saved W-ICA epochs to: {outp}")
    return cleaned_epochs


# ------------------------------ Full cleaning ------------------------------
def clean_eeg(subject_id, task_id, config=None, save_dir=None):
    if config is None:
        raise ValueError("Config must be provided.")
    save_plot_dir = config.get("save_plot_dir", "outputs")
    os.makedirs(save_plot_dir, exist_ok=True)
    filepath = get_eeg_filepath(subject_id, task_id, config)
    print(f"Processing: {filepath}")

    # READ once & cache
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
            raw.pick(keep)
        else:
            raw.pick([ch for ch in raw.ch_names if ch in montage.ch_names])
        if "l_freq" in config and "h_freq" in config:
            raw.filter(l_freq=float(config["l_freq"]), h_freq=float(config["h_freq"]))
        config["_raw_cache"] = raw

    # 1) Line-noise suppression
    raw_clean = suppress_line_noise_multitaper_and_clean(subject_id, task_id, config)

    # 2) Epochs (cache)
    epochs = config.get("_epochs_cache")
    epoch_key = (filepath, float(config["epoch_duration"]))
    if epochs is None or config.get("_epochs_key") != epoch_key or epochs._raw is not raw_clean:
        epochs = mne.make_fixed_length_epochs(raw_clean, duration=float(config["epoch_duration"]), preload=True)
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
        dpi = int(config.get("dpi", 200))
        try:
            fig_bad = epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6), show=False)
            fig_bad.suptitle(f"Bad Epochs - Subject {subject_id} Task {task_id}")
            fig_path = os.path.join(save_plot_dir, f"sub-{subject_id}_task-{task_id}_bad_epochs.png")
            fig_bad.savefig(fig_path, dpi=dpi, bbox_inches="tight"); plt.close(fig_bad)
            print(f"Saved bad epoch plot to {fig_path}")
        except Exception as e:
            print(f"Could not plot bad epochs: {e}")
        try:
            reject_log.plot("horizontal"); plt.close()
        except Exception:
            pass

    # 4) ICA (Picard) — filter Raw 1–100 Hz -> epoch -> average reference
    raw_for_ica = raw_clean.copy().filter(1.0, 100.0, fir_design="firwin", phase="zero")
    epochs_for_ica = mne.make_fixed_length_epochs(raw_for_ica, duration=float(config["epoch_duration"]), preload=True)
    epochs_for_ica.set_eeg_reference("average")

    n_comp = min(20, len(epochs_for_ica.ch_names))  # keep 20 if possible
    ica = ICA(
        n_components=n_comp,
        method="picard",
        random_state=int(config.get("random_state", 97)),
        fit_params=dict(extended=True, ortho=False),
        max_iter="auto",
    )
    ica.fit(epochs_for_ica)

    # 5) INTERACTIVE manual selection: click topomap -> properties -> toggle "Exclude"
    print("\nOpening ICA component topomaps...")
    print("Tip: Click a component to open its properties; in the properties window, toggle 'Exclude'.")
    print("Close all ICA figures when you're done — processing will continue.\n")

    figs = ica.plot_components(inst=epochs_for_ica, show=True)  # interactive
    # Block until user closes all figures
    plt.show(block=True)

    # After the GUI interaction, MNE stores your choices in ica.exclude:
    chosen = sorted(getattr(ica, "exclude", []))
    print(f"ICs marked for exclusion in GUI: {chosen}")

    # 6) Apply ICA (actual removal) + confirmation print
    raw_clean = ica.apply(raw_clean.copy())
    epochs_clean = ica.apply(epochs_clean.copy())
    print(f"Removed ICs (applied): {sorted(ica.exclude)}")

    # 7) Evoked & save diagnostic
    evoked = epochs_clean.average()
    print("Evoked shape:", evoked.data.shape)
    evoked_to_plot = evoked.copy(); evoked_to_plot.data *= 1e6
    fig_evoked = evoked_to_plot.plot(scalings=dict(eeg=20), time_unit="s", show=False)
    fig_evoked.suptitle(f"Evoked - Subject {subject_id} Task {task_id}")
    fig_ev_path = os.path.join(save_plot_dir, f"sub-{subject_id}_task-{task_id}_evoked.png")
    try:
        fig_evoked.savefig(fig_ev_path, dpi=int(config.get("dpi", 200)), bbox_inches="tight")
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
            bad_clean_path = os.path.join(save_plot_dir, f"sub-{subject_id}_task-{task_id}_bad_vs_clean.png")
            plt.gcf().savefig(bad_clean_path, dpi=int(config.get("dpi", 200)), bbox_inches="tight")
            plt.close()
            print(f"Saved bad vs clean plot to {bad_clean_path}")
        except Exception as e:
            print(f"Could not create bad vs clean plot: {e}")

    return raw_clean, epochs_clean


# ----------------------- Batch over subjects / tasks -----------------------
def preprocess_all_subjects_tasks(subjects, tasks, config=None, save_dir=None):
    for subject_id in subjects:
        for task_id in tasks:
            raw_clean, epochs_clean = clean_eeg(subject_id, task_id, config=config)
            os.makedirs(config["processed_dir"], exist_ok=True)
            path = f"{config['processed_dir']}/sub-{subject_id}_task-{task_id}_cleaned_raw.fif"
            raw_clean.save(path, overwrite=True)
            print(f"Saved cleaned data to: {path}")
