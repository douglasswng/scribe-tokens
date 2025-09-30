from constants import TMP_DIR
from utils.clear_folder import clear_folder
from utils.distributed_context import distributed_context

if distributed_context.is_master:
    clear_folder(TMP_DIR)
