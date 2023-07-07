import os

PERMDIR = os.environ.get("REFACT_PERM_DIR", "") or os.path.expanduser("~/.refact/perm-storage")

DIR_CONFIG     = os.path.join(PERMDIR, "cfg")
DIR_WATCHDOG_D = os.path.join(PERMDIR, "cfg", "watchdog.d")
DIR_WEIGHTS    = os.path.join(PERMDIR, "weights")
DIR_LOGS       = os.path.join(PERMDIR, "logs")
DIR_SSH_KEYS   = os.path.join(PERMDIR, "ssh-keys")

CONFIG_ENUM_GPUS = os.path.join(DIR_CONFIG, "gpus_enum_result.out")
CONFIG_BUSY_GPUS = os.path.join(DIR_CONFIG, "gpus_busy_result.out")
CONFIG_INFERENCE = os.path.join(DIR_CONFIG, "inference.cfg")

os.makedirs(DIR_WATCHDOG_D, exist_ok=True)
os.makedirs(DIR_WEIGHTS, exist_ok=True)
os.makedirs(DIR_LOGS, exist_ok=True)
os.makedirs(DIR_SSH_KEYS, exist_ok=True)

DIR_WATCHDOG_TEMPLATES = os.path.join(os.path.dirname(__file__), "..", "watchdog", "watchdog.d")

CHATGPT_CONFIG_FILENAME = os.path.join(DIR_CONFIG, "openai.cfg")

private_key_ext = 'private_key'
fingerprint_ext = 'fingerprint'


def get_all_ssh_keys():
    import glob
    return glob.glob(f'{DIR_SSH_KEYS}/*.{private_key_ext}')
