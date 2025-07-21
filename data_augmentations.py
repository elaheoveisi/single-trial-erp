 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, lfilter


class DataAugmentations:
    
    def __init__(self, x):
        self.x = x.astype(np.float32)
        
    def temporal_cutout(self, start, w): 
        x_temp = self.x.copy()  # Create a separate copy
        x_temp[:, start:start + w] = 0
        return x_temp.astype(np.float32)
    
    def add_noise(self, scale, same_noise=True):
        if same_noise:
            noise = np.random.randn(1, self.x.shape[1]) * scale   # scaling the noise to control noise strength
            noise = np.repeat(noise, self.x.shape[0], axis=0)     # repeats same noise for all channels
        else:
            noise = np.random.randn(*self.x.shape) * scale
        return self.x + noise

    def scaling(self, scale):
        return (self.x * scale)

    def temporal_delay(self, delay):
        delayed = np.zeros_like(self.x)
        if delay == 0:
            return self.x.copy().astype(np.float32)
        elif delay < self.x.shape[1]:
            delayed[:, delay:] = self.x[:, :-delay]
         
        return delayed.astype(np.float32)
    
 
    def channel_dropout(self, drop_idx):
        """drop_indices: list or array of channel indices to drop (e.g., [0, 3, 7] """
        C, T = self.x.shape
        mask = np.ones(C, dtype=np.float32)
        mask[drop_idx] = 0.0
        return (self.x * mask[:, None]).astype(np.float32)

    """Shift EEG channels (spatial permutation)."""
    def spatial_shift(self, shift):
        return np.roll(self.x, shift, axis=0).astype(np.float32)

    """Applies a random rotation matrix to simulate electrode space distortion."""
    def spatial_rotation(self, coordinates, alpha, beta, gamma, degrees=True ):
        """
        Rotate EEG channel xyz positions using Euler angles.
        
        Parameters:
            coords: np.ndarray of shape (C, 3) — each row is [x, y, z]
            alpha, beta, gamma: Euler angles
            degrees: whether angles are in degrees
            
        Returns:
            rotated_coords: np.ndarray of shape (C, 3)
    
        """
        if degrees:
            alpha = np.deg2rad(alpha)
            beta = np.deg2rad(beta)
            gamma = np.deg2rad(gamma)
    
        # Rotation around X-axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])
    
        # Rotation around Y-axis
        Ry = np.array([
            [ np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
    
        # Rotation around Z-axis
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [0, 0, 1]
        ])
    
        R = Rz * Ry * Rx # Combined rotation
         
        rotated_coordinates = coordinates @ R.T  
        return rotated_coordinates.astype(np.float32)
        
 
    """Zero out a contiguous block of channels (simulating spatially localized sensor dropout)."""
    def channel_cutout(self, start, h): 
        self.x[start:start + h, :] = 0
        return self.x.astype(np.float32)
    
    def signal_mixing(self, alpha, adjacency_dict, channel_names):
        """
        Mix EEG signal with its spatial neighbors based on adjacency.
    
        Parameters:
            x (ndarray): EEG signal of shape (C, T)
            alpha (float): Max mixing factor (random a ∈ [0, alpha])
            adjacency_dict (dict): Mapping of channel name to list of neighbors
            channel_names (list): List of channel names in order of x
    
        Returns:
            mixed (ndarray): Locally mixed EEG signal of shape (C, T)
        """
        name_to_idx = {ch: i for i, ch in enumerate(channel_names)}
        mixed = self.x.copy()
    
        for ch, neighbors in adjacency_dict.items():
            i = name_to_idx[ch]
            for neigh in neighbors:
                j = name_to_idx[neigh]
                a = np.random.uniform(0, alpha)
                mixed[i] = a * self.x[i] + (1 - a) * self.x[j]
    
        return mixed.astype(np.float32)
    
    def bandstop_filter(self, freq, Q, fs):
        # freq ----to be removed (central freq of notch filter)
        # quality factor , Q = freq/bandwidth , determines how wide/narrow the bandwidth(higher Q -> narrower BW)
        # fs-- sampling frequency   
        b, a = iirnotch(freq, Q, fs)   # b = numerator(ndarray), a = denominator (ndarray)  iirnotch designs the filter and provide filter coefficents
        return np.array([lfilter(b, a, ch) for ch in self.x]).astype(np.float32) # lfilter = low pass filter , applies the filter to the signal
    

#%%

''' Testing Purpose only'''

channel_names = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

adjacency_dict = {
    'Fp1': ['F7', 'F3'],
    'Fp2': ['F8', 'F4'],
    'F7':  ['Fp1', 'F3', 'T3'],
    'F3':  ['Fp1', 'F7', 'Fz', 'C3'],
    'Fz':  ['F3', 'F4', 'Cz'],
    'F4':  ['Fp2', 'Fz', 'F8', 'C4'],
    'F8':  ['Fp2', 'F4', 'T4'],

    'T3':  ['F7', 'C3', 'T5'],
    'C3':  ['F3', 'T3', 'Cz', 'P3'],
    'Cz':  ['Fz', 'C3', 'C4', 'Pz'],
    'C4':  ['F4', 'Cz', 'T4', 'P4'],
    'T4':  ['F8', 'C4', 'T6'],

    'T5':  ['T3', 'P3', 'O1'],
    'P3':  ['T5', 'C3', 'Pz'],
    'Pz':  ['P3', 'Cz', 'P4'],
    'P4':  ['Pz', 'C4', 'T6'],
    'T6':  ['T4', 'P4', 'O2'],

    'O1':  ['T5', 'O2'],
    'O2':  ['T6', 'O1']
}

C, T = len(channel_names), 300
t = np.linspace(0, 1, T)
toy_signal = np.sin(2 * np.pi * 10 * t)[None, :] * np.linspace(1, 0.5, C)[:, None]

# Simulated EEG electrode coordinates (e.g., 5 electrodes)
coordinates = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.5],
    [0.0, 1.0, 0.5],
    [-1.0, 0.0, 0.5],
    [0.0, -1.0, 0.5]
])

augmentations = DataAugmentations(toy_signal.copy())
 
augmented = {
    "Original": toy_signal,
    "Temporal_Cutout": augmentations.temporal_cutout(100, 50),
    "Noise": augmentations.add_noise(0.3),
    "Scaled": augmentations.scaling(0.6),
    "Temporal_Delay": augmentations.temporal_delay(20), 
    "Spatial_Shift": augmentations.spatial_shift(2), 
    "Dropout": augmentations.channel_dropout([7,8,17,18]),
    "Mixed":augmentations.signal_mixing(0.5, adjacency_dict, channel_names),
    "Channel_Cutout": augmentations.channel_cutout(7,2),
    "Bandstop_Filter": augmentations.bandstop_filter(10, 0.1, 300)
}



# Rotate by 30° around X, 45° around Y, and 60° around Z
rotated_coordinates = augmentations.spatial_rotation(coordinates, alpha=30, beta=45, gamma=60)

print("Original:\n", coordinates)
print("Rotated:\n", rotated_coordinates)


plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20        
plt.rcParams['axes.titleweight'] = 'bold'  
channels_to_plot = [0, 8, 17]  # Fp1, C3, O1

for label, signal in augmented.items():
    if label == "Original":
        continue

    fig, axs = plt.subplots(len(channels_to_plot), 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"Original vs {label}", fontsize=20, fontweight='bold')

    for idx, ch in enumerate(channels_to_plot):
        axs[idx].plot(augmented["Original"][ch], label=f'Original - {channel_names[ch]}', linewidth=1.5)
        axs[idx].plot(signal[ch], label=f'{label} - {channel_names[ch]}', linestyle='--', linewidth=1.5)
        axs[idx].legend(loc='upper right')
        axs[idx].set_ylabel("Amplitude")
        axs[idx].set_ylim(-1, 1)  # Set y-axis limits
        axs[idx].grid(True)

    axs[-1].set_xlabel("Time (samples)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()