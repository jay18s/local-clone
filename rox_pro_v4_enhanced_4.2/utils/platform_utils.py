"""
ROX Proven Edge Engine v3.0 - Windows-Compatible Utilities
=========================================================
Platform-independent utility functions for file handling, paths, and system operations.
"""

import os
import sys
import platform
import shutil
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio


logger = logging.getLogger("PlatformUtils")


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system().lower() == "windows"


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system().lower() == "linux"


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system().lower() == "darwin"


def get_platform_name() -> str:
    """Get the platform name."""
    return platform.system()


def get_path(*parts: str) -> Path:
    """
    Create a platform-independent path.
    
    Args:
        *parts: Path components to join
        
    Returns:
        Path object for the given components
    """
    return Path(*parts)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path object for the directory
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_app_data_dir() -> Path:
    """
    Get application data directory in a platform-independent way.
    
    Returns:
        Path to application data directory
    """
    if is_windows():
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    elif is_macos():
        base = Path.home() / 'Library' / 'Application Support'
    else:  # Linux and others
        base = Path.home() / '.local' / 'share'
    
    app_dir = base / 'ROXEdgeEngine'
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_log_dir() -> Path:
    """Get the log directory path."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent.parent
    
    log_dir = base / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_data_dir() -> Path:
    """Get the data directory path."""
    if getattr(sys, 'frozen', False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent.parent
    
    data_dir = base / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def safe_read_file(filepath: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
    """
    Safely read a file with proper error handling.
    
    Args:
        filepath: Path to the file
        encoding: File encoding (default: utf-8)
        
    Returns:
        File contents as string, or None if error
    """
    try:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        with open(path, 'r', encoding=encoding, newline='') as f:
            return f.read()
    except PermissionError:
        logger.error(f"Permission denied: {filepath}")
        return None
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return None
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None


def safe_write_file(filepath: Union[str, Path], content: str, 
                   encoding: str = 'utf-8', backup: bool = False) -> bool:
    """
    Safely write to a file with proper error handling.
    
    Args:
        filepath: Path to the file
        content: Content to write
        encoding: File encoding (default: utf-8)
        backup: Whether to create a backup before overwriting
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(filepath)
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file if requested
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + '.bak')
            shutil.copy2(path, backup_path)
        
        # Write with proper line endings for the platform
        with open(path, 'w', encoding=encoding, newline='') as f:
            f.write(content)
        
        return True
    except PermissionError:
        logger.error(f"Permission denied: {filepath}")
        return False
    except Exception as e:
        logger.error(f"Error writing file {filepath}: {e}")
        return False


def safe_read_json(filepath: Union[str, Path]) -> Optional[Dict]:
    """
    Safely read a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Parsed JSON as dictionary, or None if error
    """
    try:
        content = safe_read_file(filepath)
        if content is None:
            return None
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading JSON file {filepath}: {e}")
        return None


def safe_write_json(filepath: Union[str, Path], data: Dict, 
                   indent: int = 2, backup: bool = False) -> bool:
    """
    Safely write a JSON file.
    
    Args:
        filepath: Path to the JSON file
        data: Data to write
        indent: JSON indentation
        backup: Whether to create a backup
        
    Returns:
        True if successful, False otherwise
    """
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
        return safe_write_file(filepath, content, backup=backup)
    except Exception as e:
        logger.error(f"Error writing JSON file {filepath}: {e}")
        return False


def safe_read_csv(filepath: Union[str, Path]) -> Optional[List[Dict]]:
    """
    Safely read a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of dictionaries, or None if error
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return []
        
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        logger.error(f"Error reading CSV file {filepath}: {e}")
        return None


def safe_write_csv(filepath: Union[str, Path], data: List[Dict], 
                  fieldnames: List[str] = None, backup: bool = False) -> bool:
    """
    Safely write a CSV file.
    
    Args:
        filepath: Path to the CSV file
        data: List of dictionaries to write
        fieldnames: Column names (auto-detected if not provided)
        backup: Whether to create a backup
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not data:
            # Write empty file with headers only
            return True
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        # Backup existing file if requested
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + '.bak')
            shutil.copy2(path, backup_path)
        
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        return True
    except Exception as e:
        logger.error(f"Error writing CSV file {filepath}: {e}")
        return False


def append_to_csv(filepath: Union[str, Path], row: Dict, 
                 fieldnames: List[str] = None) -> bool:
    """
    Append a row to a CSV file.
    
    Args:
        filepath: Path to the CSV file
        row: Row dictionary to append
        fieldnames: Column names (only needed for new file)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        file_exists = path.exists()
        
        if fieldnames is None:
            fieldnames = list(row.keys())
        
        with open(path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        
        return True
    except Exception as e:
        logger.error(f"Error appending to CSV file {filepath}: {e}")
        return False


def clean_filename(filename: str) -> str:
    """
    Clean a filename to be safe for all platforms.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Characters not allowed on Windows
    invalid_chars = '<>:"/\\|?*'
    
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Ensure not empty
    if not filename:
        filename = 'unnamed'
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def get_file_age(filepath: Union[str, Path]) -> Optional[float]:
    """
    Get the age of a file in seconds.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Age in seconds, or None if file doesn't exist
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return None
        
        mtime = path.stat().st_mtime
        return datetime.now().timestamp() - mtime
    except Exception as e:
        logger.error(f"Error getting file age for {filepath}: {e}")
        return None


def file_exists_and_recent(filepath: Union[str, Path], max_age_seconds: float) -> bool:
    """
    Check if a file exists and is recent.
    
    Args:
        filepath: Path to check
        max_age_seconds: Maximum age in seconds
        
    Returns:
        True if file exists and is recent enough
    """
    age = get_file_age(filepath)
    return age is not None and age < max_age_seconds


def delete_old_files(directory: Union[str, Path], max_age_days: int, 
                    pattern: str = '*') -> int:
    """
    Delete files older than a certain age.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0
        
        max_age_seconds = max_age_days * 24 * 60 * 60
        deleted = 0
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                age = get_file_age(file_path)
                if age and age > max_age_seconds:
                    file_path.unlink()
                    deleted += 1
        
        return deleted
    except Exception as e:
        logger.error(f"Error cleaning old files in {directory}: {e}")
        return 0


def get_disk_space(directory: Union[str, Path]) -> Dict[str, float]:
    """
    Get disk space information.
    
    Args:
        directory: Directory to check
        
    Returns:
        Dictionary with 'total', 'used', 'free' in GB
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path = dir_path.parent
        
        total, used, free = shutil.disk_usage(dir_path)
        
        return {
            'total_gb': total / (1024 ** 3),
            'used_gb': used / (1024 ** 3),
            'free_gb': free / (1024 ** 3),
            'used_percent': (used / total) * 100
        }
    except Exception as e:
        logger.error(f"Error getting disk space: {e}")
        return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'used_percent': 0}


async def run_async_command(cmd: List[str], timeout: float = 30) -> Dict:
    """
    Run a command asynchronously.
    
    Args:
        cmd: Command and arguments as list
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with 'success', 'stdout', 'stderr', 'returncode'
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )
        
        return {
            'success': process.returncode == 0,
            'stdout': stdout.decode('utf-8', errors='replace'),
            'stderr': stderr.decode('utf-8', errors='replace'),
            'returncode': process.returncode
        }
    except asyncio.TimeoutError:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Command timed out',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }


def get_system_info() -> Dict[str, str]:
    """
    Get system information.
    
    Returns:
        Dictionary with system details
    """
    return {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'node': platform.node(),
        'is_frozen': getattr(sys, 'frozen', False),
    }
