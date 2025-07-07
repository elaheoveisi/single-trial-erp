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
    raw.set_montage(montage, on_missing='ignore')  # Ignore the unmatched channels
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
    # Manually exclude specific components 
    ica.exclude = [0, 3]  

def apply_ica(ica, raw):
    ica.apply(raw)
    return raw

def create_epochs(raw, duration=0.7, baseline=(None, 0)):
    """Create fixed-length epochs from raw data."""
    return mne.make_fixed_length_epochs(raw, duration=duration, baseline=baseline, preload=True)



def validate_results(raw, epochs=None):
    """Visualize cleaned raw data and optionally averaged ERP."""
    raw.plot(scalings='auto', show=True)
    if epochs is not None:
        epochs.average().plot(title="Average ERP", show=True)

# Main function to run all steps
def run_ica_pipeline():
    raw = load_raw_data()
    raw = set_montage(raw)             
    pick_eeg_channels(raw)      
    raw = preprocess_data(raw)
    
    ica = run_ica(raw)
    visualize_components(ica, raw)
    exclude_artifacts(ica)             
    raw = apply_ica(ica, raw)
    
    epochs = create_epochs(raw, tmin=-0.2, tmax=0.5, baseline=(None, 0))
    validate_results(epochs, None)

# Run the pipeline
run_ica_pipeline()
