#Preprocessing using ICA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import time
import mne
from mne.channels import make_standard_montage
from mne.preprocessing import ICA

def load_raw_data(raw_fname=r"C:\Users\elahe\Downloads\sub-001_task-lumfront_eeg.bdf", tmin=0, tmax=60):
    """Load and crop raw EEG data."""
    raw = mne.io.read_raw_bdf(raw_fname, preload=True).crop(tmin, tmax)
    return raw

def set_montage(raw):
    """Apply a standard montage for channel locations, ignoring missing channels."""
    montage = make_standard_montage('standard_1005')  # 10-10 system for 72 channels
    raw.set_montage(montage, on_missing='ignore')  # Ignore unmapped EXG channels
    return raw

def pick_eeg_channels(raw):
    """Select only EEG channels, excluding EXG."""
    raw.pick([ch for ch in raw.ch_names if not ch.startswith('EXG')])
    return raw


# Step 2: Preprocess the data
def preprocess_data(raw):
    raw.filter(l_freq=1.0, h_freq=60.0)
    return raw

# Step 3: Fit the ICA model and visualize

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
def visualize_components(ica, raw):
    ica.plot_components(inst=raw)

def exclude_artifacts(ica):
    # Manually exclude specific components (e.g., ICA001 and ICA014)
    ica.exclude = [0, 3]  # Replace with the indices of artifact-related components

def apply_ica(ica, raw):
    ica.apply(raw)
    return raw

def create_epochs(raw, event_id=None, tmin=-0.2, tmax=0.5, baseline=(None, 0)):
    """Extract events and create epochs from raw data."""
    # Find events (assuming events are stored in a trigger channel or annotations)
    events, event_id = mne.events_from_annotations(raw, event_id=event_id)
    if len(events) == 0:
        raise ValueError("No events found in the data. Check trigger channel or event codes.")
    
    # Create epochs
    epochs = mne.preprocessing.Epochs(
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=raw,
            baseline=baseline,
            preload=True,
            reject=None
        )
    )
    return epochs

def validate_results(raw, epochs=None):
    """Visualize cleaned raw data and optionally averaged ERP."""
    raw.plot(scalings='auto', show=True)
    if epochs is not None:
        epochs.average().plot(title="Average ERP", show=True)

# Main function to run all steps
def run_ica_pipeline():
    raw = load_raw_data()
    raw = set_montage(raw)             # <--- Add this
    raw = pick_eeg_channels(raw)       # <--- Add this
    raw = preprocess_data(raw)
    ica = run_ica(raw)
    visualize_components(ica, raw)
    exclude_artifacts(ica)             # <--- Fix: remove `=`, it returns None
    raw = apply_ica(ica, raw)
    # Optional: Add event extraction here if needed
    # epochs = create_epochs(raw)
    validate_results(raw, None)        # You can pass `epochs` here if available

# Run the pipeline
run_ica_pipeline()
