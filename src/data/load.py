import openneuro


def download_data(subjects, tasks, config=None):
    included_files = []
    for subj in subjects:
        for task in tasks:
            subject_file = f"sub-{subj}/eeg/sub-{subj}_task-{task}_eeg.bdf"
            included_files.append(subject_file)

        openneuro.download(
            dataset=config["dataset"],
            target_dir=config["target_dir"],
            include=included_files,
        )
