def plot_psd(raw, fmax=40):
    """
    Parameters:
    - raw: mne.io.Raw
        The raw EEG data object.
    - fmax: float, optional (default=250)
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
    return fig


