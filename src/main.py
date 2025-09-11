import yaml
from utils import skip_run

from data.load import download_data
from data.preprocessing import clean_eeg, preprocess_all_subjects_tasks

# The configuration file
config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("skip", "download_data") as check, check():
    download_data(config["subject_ids"], tasks=config["tasks"], config=config)


with skip_run("skip", "clean_example_data") as check, check():
    subjects = config["subject_ids"]
    tasks = config["tasks"]
    # Use task_id (not "tasks") to match clean_eeg signature
    clean_eeg(subject_id=subjects[0], task_id=tasks[0], config=config)


with skip_run("run", "clean_all_data") as check, check():
    preprocess_all_subjects_tasks(
        config["subject_ids"], tasks=config["tasks"], config=config
    )
