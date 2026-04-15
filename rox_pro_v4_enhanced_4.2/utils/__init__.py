"""
ROX Proven Edge Engine v3.0 - Utilities Package
=============================================
Helper functions and utilities for cross-platform compatibility.
"""

from .helpers import (
    format_currency,
    format_percentage,
    calculate_risk_reward,
    calculate_shares,
    validate_price,
    normalize_score
)

from .platform_utils import (
    is_windows,
    is_linux,
    is_macos,
    get_platform_name,
    get_path,
    ensure_dir,
    get_app_data_dir,
    get_log_dir,
    get_data_dir,
    safe_read_file,
    safe_write_file,
    safe_read_json,
    safe_write_json,
    safe_read_csv,
    safe_write_csv,
    append_to_csv,
    clean_filename,
    get_file_age,
    file_exists_and_recent,
    delete_old_files,
    get_disk_space,
    get_system_info,
)

__all__ = [
    # Helpers
    "format_currency",
    "format_percentage",
    "calculate_risk_reward",
    "calculate_shares",
    "validate_price",
    "normalize_score",
    # Platform utils
    "is_windows",
    "is_linux",
    "is_macos",
    "get_platform_name",
    "get_path",
    "ensure_dir",
    "get_app_data_dir",
    "get_log_dir",
    "get_data_dir",
    "safe_read_file",
    "safe_write_file",
    "safe_read_json",
    "safe_write_json",
    "safe_read_csv",
    "safe_write_csv",
    "append_to_csv",
    "clean_filename",
    "get_file_age",
    "file_exists_and_recent",
    "delete_old_files",
    "get_disk_space",
    "get_system_info",
]
