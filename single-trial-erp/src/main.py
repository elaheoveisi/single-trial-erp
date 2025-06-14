import yaml

from utils import skip_run

# The configuration file
config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("skip", "crop_and_replot") as check, check():
    pass  