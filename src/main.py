import yaml
from utils import skip_run
import os
import mne
from mne.preprocessing import ICA
import pywt
import numpy as np
import openneuro


from data.load import download_data
from data.preprocessing import  load_eeg_data,suppress_line_noise_multitaper_and_clean,wavelet_denoise_epochs_from_epochs, clean_eeg, preprocess_all_subjects_tasks, wavelet_denoise_epochs_from_epochs,get_eeg_filepath

# The configuration file
config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

# Create save_dir folder if it doesn't exist
save_dir = os.path.join(os.getcwd(), config.get("save_dir", "outputs"))
os.makedirs(save_dir, exist_ok=True)

# Optional: update config with full path
config["save_dir"] = save_dir





with skip_run("skip", "download_data") as check, check():
    download_data(config["subjects"], tasks=config["tasks"], config=config)

            
                

subject_id = config["subjects"][0]
task_id = config["tasks"][0]
filepath = get_eeg_filepath(subject_id, task_id, config)


with skip_run("skip", "load eeg") as check:
    if check():
        subjects = config["subjects"]
        tasks = config["tasks"]
        load_eeg_data(subject_id=subjects[0], task_id=tasks[0], config=config)


with skip_run("skip", "load_eeg") as check:
    if check():
        subjects = config["subjects"]
        tasks = config["tasks"]

        raw , epochs = load_eeg_data(
            subject_id=subjects[0],
            task_id=tasks[0],
            config=config
        )



with skip_run("skip", "suppress_line_noise") as check:
    if check():
        epochs_clean = suppress_line_noise_multitaper_and_clean(
            subject_id=subjects[0],
            task_id=tasks[0],
            config=config
        )


with skip_run("skip", "wavelet") as check:
    if check():
        subjects = config["subjects"]
        tasks = config["tasks"]
        wavelet_denoise_epochs_from_epochs(subject_id=subjects[0], task_id=tasks[0], config=config)





with skip_run("run", "clean_example_data") as check:
    if check():
        subjects = config["subjects"]
        tasks = config["tasks"]
        clean_eeg(subject_id=subjects[0], task_id=tasks[0], config=config)



with skip_run("skip", "clean_all_data") as check, check():
    preprocess_all_subjects_tasks(
        subjects=config["subjects"],
        tasks=config["tasks"],
        config=config
    )


