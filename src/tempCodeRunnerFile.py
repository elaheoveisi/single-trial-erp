with skip_run("run", "clean_all_data") as check, check():
    preprocess_all_subjects_tasks(config["subjects"], tasks=config["tasks"], config=config)