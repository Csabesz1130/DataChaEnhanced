"""
Simplified Set Exporter - Redirects to batch_set_exporter
Location: src/gui/simplified_set_exporter.py
This file exists for backward compatibility
"""

# Import everything from the batch set exporter
from .batch_set_exporter import (
    BatchSetExporter,
    BatchSetExportButton,
    add_set_export_to_toolbar
)

# For backward compatibility, create aliases
SimplifiedSetExporter = BatchSetExporter
SimplifiedSetExportButton = BatchSetExportButton

__all__ = [
    'BatchSetExporter',
    'BatchSetExportButton', 
    'add_set_export_to_toolbar',
    'SimplifiedSetExporter',
    'SimplifiedSetExportButton'
]