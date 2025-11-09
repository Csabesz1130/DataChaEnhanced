"""
Export API endpoints
Generate Excel/CSV exports using desktop code
"""

from flask import Blueprint, request, jsonify, current_app, send_file
import os
import tempfile
from pathlib import Path

# Import desktop export code
# Note: excel_export uses tkinter dialogs, so we'll implement web-specific export
# from src.excel_export.excel_export import export_to_excel
from src.csv_export.dual_curves_csv_export import export_dual_curves_to_csv

from backend.utils.db import db, AnalysisResult, ExportFile

bp = Blueprint('export', __name__)


@bp.route('/excel', methods=['POST'])
def export_to_excel():
    """
    Generate Excel export with charts
    
    Request JSON:
    {
        "analysis_id": "uuid",
        "include_charts": true,
        "include_raw_data": true
    }
    
    Returns:
    {
        "export_id": "uuid",
        "download_url": "/api/export/download/<export_id>",
        "filename": "analysis_results.xlsx"
    }
    """
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        include_charts = data.get('include_charts', True)
        include_raw_data = data.get('include_raw_data', True)
        
        if not analysis_id:
            return jsonify({'error': 'analysis_id is required'}), 400
        
        # Get analysis results
        analysis = AnalysisResult.query.get(analysis_id)
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.xlsx',
            dir=current_app.config['UPLOAD_FOLDER']
        )
        temp_file.close()
        
        # Generate Excel using desktop code
        # Note: May need to adapt ExcelExporter to work without GUI
        # For now, implement basic export
        import pandas as pd
        from openpyxl import Workbook
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Analysis Results"
        
        # Add parameters
        ws.append(['Parameter', 'Value'])
        for key, value in analysis.params.items():
            ws.append([key, value])
        
        ws.append([])  # Empty row
        
        # Add results summary
        ws.append(['Result', 'Value'])
        results = analysis.results
        ws.append(['Baseline', results.get('baseline')])
        ws.append(['Number of Cycles', results.get('cycles')])
        
        # Add curve data if requested
        if include_raw_data:
            ws.append([])
            ws.append(['Orange Curve'])
            ws.append(['Time (s)', 'Current (pA)'])
            
            orange_times = results.get('orange_curve_times', [])
            orange_curve = results.get('orange_curve', [])
            
            for t, c in zip(orange_times[:1000], orange_curve[:1000]):  # Limit to 1000 points
                ws.append([t, c])
        
        wb.save(temp_file.name)
        
        # Save export record to database
        export_file = ExportFile(
            analysis_id=analysis_id,
            export_type='excel',
            file_path=temp_file.name,
            filename='analysis_results.xlsx'
        )
        db.session.add(export_file)
        db.session.commit()
        
        current_app.logger.info(f"Excel export created: {export_file.id}")
        
        return jsonify({
            'export_id': str(export_file.id),
            'download_url': f'/api/export/download/{export_file.id}',
            'filename': 'analysis_results.xlsx'
        }), 201
        
    except Exception as e:
        current_app.logger.error(f"Excel export error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Export failed', 'message': str(e)}), 500


@bp.route('/csv', methods=['POST'])
def export_to_csv():
    """
    Generate CSV export
    
    Request JSON:
    {
        "analysis_id": "uuid"
    }
    
    Returns:
    {
        "export_id": "uuid",
        "download_url": "/api/export/download/<export_id>",
        "filename": "analysis_results.csv"
    }
    """
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        
        if not analysis_id:
            return jsonify({'error': 'analysis_id is required'}), 400
        
        # Get analysis results
        analysis = AnalysisResult.query.get(analysis_id)
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.csv',
            dir=current_app.config['UPLOAD_FOLDER']
        )
        temp_file.close()
        
        # Generate CSV
        import csv
        
        with open(temp_file.name, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write parameters
            writer.writerow(['Parameters'])
            for key, value in analysis.params.items():
                writer.writerow([key, value])
            
            writer.writerow([])  # Empty row
            
            # Write curve data
            results = analysis.results
            writer.writerow(['Orange Curve Data'])
            writer.writerow(['Time (s)', 'Current (pA)'])
            
            orange_times = results.get('orange_curve_times', [])
            orange_curve = results.get('orange_curve', [])
            
            for t, c in zip(orange_times, orange_curve):
                writer.writerow([t, c])
        
        # Save export record
        export_file = ExportFile(
            analysis_id=analysis_id,
            export_type='csv',
            file_path=temp_file.name,
            filename='analysis_results.csv'
        )
        db.session.add(export_file)
        db.session.commit()
        
        current_app.logger.info(f"CSV export created: {export_file.id}")
        
        return jsonify({
            'export_id': str(export_file.id),
            'download_url': f'/api/export/download/{export_file.id}',
            'filename': 'analysis_results.csv'
        }), 201
        
    except Exception as e:
        current_app.logger.error(f"CSV export error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Export failed', 'message': str(e)}), 500


@bp.route('/download/<export_id>', methods=['GET'])
def download_export(export_id):
    """Download generated export file"""
    try:
        export = ExportFile.query.get(export_id)
        if not export:
            return jsonify({'error': 'Export not found'}), 404
        
        if not os.path.exists(export.file_path):
            return jsonify({'error': 'Export file no longer exists'}), 404
        
        return send_file(
            export.file_path,
            as_attachment=True,
            download_name=export.filename
        )
        
    except Exception as e:
        current_app.logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500

