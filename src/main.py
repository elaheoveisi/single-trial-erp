import yaml
from utils import skip_run
import os

from data.load import download_data
from data.preprocessing import clean_eeg, preprocess_all_subjects_tasks
from data.powerline import plot_psd

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


with skip_run("skip", "clean_example_data") as check, check():
    subjects = config["subjects"]
    tasks = config["tasks"]
    clean_eeg(subject_id=subjects[0], task_id=tasks[0], config=config)

with skip_run("skip", "clean_all_data") as check, check():
    preprocess_all_subjects_tasks(
        subjects=config["subjects"],
        tasks=config["tasks"],
        config=config
    )


with skip_run("run", "plot_psd") as check, check():
    plot_psd(
        config["subjects"], tasks=config["tasks"], config=config
    )