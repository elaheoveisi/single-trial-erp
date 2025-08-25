with skip_run("run", "wavelet") as check:
    if check():
        subjects = config["subjects"]
        tasks = config["tasks"]
        wavelet_denoise_epochs_from_epochs(subject_id=subjects[0], task_id=tasks[0], config=config)

