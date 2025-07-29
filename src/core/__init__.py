from core.utils.clear_folder import clear_folder
from core.utils.distributed_context import distributed_context
from core.constants import TMP_DIR


if distributed_context.is_master:
    clear_folder(TMP_DIR, confirm=False)