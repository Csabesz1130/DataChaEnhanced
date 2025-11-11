"""
Local cache using TinyDB for fast lookups
Optional edge cache for analysis runs
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from tinydb import TinyDB, Query
    TINYDB_AVAILABLE = True
except ImportError:
    TINYDB_AVAILABLE = False

# Cache file location
CACHE_DIR = Path(os.environ.get('CACHE_DIR', '/tmp'))
CACHE_FILE = CACHE_DIR / 'analysis_cache.json'

_db = None


def get_cache_db():
    """Get or create TinyDB instance"""
    global _db
    
    if not TINYDB_AVAILABLE:
        return None
    
    if _db is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _db = TinyDB(str(CACHE_FILE))
    
    return _db


def cache_run(run_id: str, run_data: Dict[str, Any]) -> bool:
    """
    Cache an analysis run in TinyDB
    
    Args:
        run_id: Run ID
        run_data: Run data dictionary
        
    Returns:
        bool: True if cached successfully
    """
    db = get_cache_db()
    if not db:
        return False
    
    try:
        runs_table = db.table('runs')
        run_data['_id'] = run_id
        runs_table.upsert(run_data, Query()._id == run_id)
        return True
    except Exception as e:
        print(f"Error caching run: {e}")
        return False


def get_cached_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a cached analysis run from TinyDB
    
    Args:
        run_id: Run ID
        
    Returns:
        dict or None: Cached run data or None if not found
    """
    db = get_cache_db()
    if not db:
        return None
    
    try:
        runs_table = db.table('runs')
        result = runs_table.search(Query()._id == run_id)
        if result:
            data = result[0].copy()
            data.pop('_id', None)  # Remove internal ID
            return data
        return None
    except Exception as e:
        print(f"Error getting cached run: {e}")
        return None


def clear_cache() -> bool:
    """Clear all cached runs"""
    db = get_cache_db()
    if not db:
        return False
    
    try:
        db.drop_table('runs')
        return True
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False


