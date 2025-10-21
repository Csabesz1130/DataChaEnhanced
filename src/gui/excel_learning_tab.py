#!/usr/bin/env python3
"""
Excel Learning Background Task Tab

This module provides a GUI tab for managing Excel file learning tasks in the background.
Users can submit Excel files for AI learning, monitor progress, and view results.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from src.utils.logger import app_logger


class ExcelLearningTab:
    """Tab for managing Excel learning background tasks"""

    def __init__(self, notebook, app):
        """Initialize the Excel Learning tab"""
        self.notebook = notebook
        self.app = app
        self.frame = ttk.Frame(notebook)

        # Initialize background processor
        self.background_processor = None
        self.initialize_background_processor()

        # Task tracking
        self.active_tasks = {}
        self.task_widgets = {}

        # UI update thread
        self.update_thread = None
        self.stop_updates = False

        # Setup UI
        self.setup_ui()

        # Start UI updates
        self.start_ui_updates()

        app_logger.info("Excel Learning tab initialized")

    def initialize_background_processor(self):
        """Initialize the background processor with error handling"""
        try:
            # Try the simplified background processor first (no heavy dependencies)
            from src.ai_excel_learning.background_processor_simple import (
                SimpleBackgroundProcessor,
                TaskStatus,
                NotificationType,
            )

            # Create simplified background processor
            self.background_processor = SimpleBackgroundProcessor(
                max_workers=2,
                storage_path="excel_learning_data",
                enable_notifications=True,
            )

            # Add notification handler
            self.background_processor.add_notification_handler(self.on_notification)

            # Start processing
            self.background_processor.start_processing()

            app_logger.info("Simplified background processor initialized successfully")

        except ImportError as e:
            app_logger.warning(f"Could not import simplified background processor: {e}")
            # Try the full background processor as fallback
            try:
                from src.ai_excel_learning.background_processor import (
                    BackgroundProcessor,
                )
                from src.ai_excel_learning import TaskStatus, NotificationType

                # Create background processor
                self.background_processor = BackgroundProcessor(
                    max_workers=2,
                    storage_path="excel_learning_data",
                    enable_notifications=True,
                )

                # Add notification handler
                self.background_processor.add_notification_handler(self.on_notification)

                # Start processing
                self.background_processor.start_processing()

                app_logger.info("Full background processor initialized successfully")

            except ImportError as e2:
                app_logger.warning(f"Could not import full background processor: {e2}")
                self.background_processor = None
            except Exception as e2:
                app_logger.error(f"Error initializing full background processor: {e2}")
                self.background_processor = None
        except Exception as e:
            app_logger.error(f"Error initializing simplified background processor: {e}")
            self.background_processor = None

    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Excel AI Learning Background Tasks",
            font=("Arial", 14, "bold"),
        )
        title_label.pack(pady=(0, 20))

        # Create notebook for different sections
        self.tab_notebook = ttk.Notebook(main_frame)
        self.tab_notebook.pack(fill="both", expand=True)

        # Submit Task Tab
        self.setup_submit_tab()

        # Monitor Tasks Tab
        self.setup_monitor_tab()

        # Results Tab
        self.setup_results_tab()

        # Status bar
        self.setup_status_bar(main_frame)

    def setup_submit_tab(self):
        """Setup the task submission tab"""
        submit_frame = ttk.Frame(self.tab_notebook)
        self.tab_notebook.add(submit_frame, text="Submit Task")

        # File selection section
        file_frame = ttk.LabelFrame(
            submit_frame, text="Excel File Selection", padding=10
        )
        file_frame.pack(fill="x", padx=10, pady=10)

        # File path
        ttk.Label(file_frame, text="Excel File:").pack(anchor="w")

        file_path_frame = ttk.Frame(file_frame)
        file_path_frame.pack(fill="x", pady=5)

        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(
            file_path_frame, textvariable=self.file_path_var, width=50
        )
        self.file_path_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(file_path_frame, text="Browse", command=self.browse_file).pack(
            side="right", padx=(5, 0)
        )

        # Task configuration section
        config_frame = ttk.LabelFrame(
            submit_frame, text="Task Configuration", padding=10
        )
        config_frame.pack(fill="x", padx=10, pady=10)

        # Task type
        ttk.Label(config_frame, text="Learning Type:").pack(anchor="w")
        self.task_type_var = tk.StringVar(value="full_pipeline")
        task_type_combo = ttk.Combobox(
            config_frame, textvariable=self.task_type_var, state="readonly"
        )
        task_type_combo["values"] = [
            "full_pipeline",
            "excel_analysis",
            "formula_learning",
            "chart_learning",
        ]
        task_type_combo.pack(fill="x", pady=5)

        # Priority
        ttk.Label(config_frame, text="Priority:").pack(anchor="w")
        self.priority_var = tk.IntVar(value=3)
        priority_frame = ttk.Frame(config_frame)
        priority_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(
            priority_frame, text="Low", variable=self.priority_var, value=1
        ).pack(side="left")
        ttk.Radiobutton(
            priority_frame, text="Medium", variable=self.priority_var, value=3
        ).pack(side="left", padx=10)
        ttk.Radiobutton(
            priority_frame, text="High", variable=self.priority_var, value=5
        ).pack(side="left")

        # Submit button
        submit_frame_buttons = ttk.Frame(submit_frame)
        submit_frame_buttons.pack(fill="x", padx=10, pady=20)

        self.submit_button = ttk.Button(
            submit_frame_buttons, text="Submit Learning Task", command=self.submit_task
        )
        self.submit_button.pack(side="left")

        ttk.Button(
            submit_frame_buttons, text="Clear", command=self.clear_submit_form
        ).pack(side="left", padx=10)

        # Status display
        self.submit_status_var = tk.StringVar()
        ttk.Label(
            submit_frame, textvariable=self.submit_status_var, foreground="blue"
        ).pack(pady=10)

    def setup_monitor_tab(self):
        """Setup the task monitoring tab"""
        monitor_frame = ttk.Frame(self.tab_notebook)
        self.tab_notebook.add(monitor_frame, text="Monitor Tasks")

        # Control buttons
        control_frame = ttk.Frame(monitor_frame)
        control_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(control_frame, text="Refresh", command=self.refresh_tasks).pack(
            side="left"
        )
        ttk.Button(
            control_frame, text="Clear Completed", command=self.clear_completed_tasks
        ).pack(side="left", padx=10)
        ttk.Button(control_frame, text="Stop All", command=self.stop_all_tasks).pack(
            side="left", padx=10
        )

        # Tasks list
        tasks_frame = ttk.LabelFrame(monitor_frame, text="Active Tasks", padding=10)
        tasks_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create treeview for tasks
        columns = ("Task ID", "File", "Type", "Status", "Progress", "Created")
        self.tasks_tree = ttk.Treeview(
            tasks_frame, columns=columns, show="headings", height=10
        )

        # Configure columns
        for col in columns:
            self.tasks_tree.heading(col, text=col)
            self.tasks_tree.column(col, width=100)

        # Scrollbar
        tasks_scrollbar = ttk.Scrollbar(
            tasks_frame, orient="vertical", command=self.tasks_tree.yview
        )
        self.tasks_tree.configure(yscrollcommand=tasks_scrollbar.set)

        self.tasks_tree.pack(side="left", fill="both", expand=True)
        tasks_scrollbar.pack(side="right", fill="y")

        # Task details
        details_frame = ttk.LabelFrame(monitor_frame, text="Task Details", padding=10)
        details_frame.pack(fill="x", padx=10, pady=10)

        self.task_details_text = tk.Text(details_frame, height=8, wrap="word")
        details_scrollbar = ttk.Scrollbar(
            details_frame, orient="vertical", command=self.task_details_text.yview
        )
        self.task_details_text.configure(yscrollcommand=details_scrollbar.set)

        self.task_details_text.pack(side="left", fill="both", expand=True)
        details_scrollbar.pack(side="right", fill="y")

        # Bind selection event
        self.tasks_tree.bind("<<TreeviewSelect>>", self.on_task_select)

    def setup_results_tab(self):
        """Setup the results viewing tab"""
        results_frame = ttk.Frame(self.tab_notebook)
        self.tab_notebook.add(results_frame, text="Results")

        # Control buttons
        results_control_frame = ttk.Frame(results_frame)
        results_control_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(
            results_control_frame, text="Refresh Results", command=self.refresh_results
        ).pack(side="left")
        ttk.Button(
            results_control_frame, text="Export Results", command=self.export_results
        ).pack(side="left", padx=10)
        ttk.Button(
            results_control_frame, text="Clear Results", command=self.clear_results
        ).pack(side="left", padx=10)

        # Results list
        results_list_frame = ttk.LabelFrame(
            results_frame, text="Completed Tasks", padding=10
        )
        results_list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create treeview for results
        result_columns = (
            "Task ID",
            "File",
            "Type",
            "Status",
            "Completed",
            "Result Size",
        )
        self.results_tree = ttk.Treeview(
            results_list_frame, columns=result_columns, show="headings", height=8
        )

        # Configure columns
        for col in result_columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)

        # Scrollbar
        results_scrollbar = ttk.Scrollbar(
            results_list_frame, orient="vertical", command=self.results_tree.yview
        )
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)

        self.results_tree.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")

        # Result details
        result_details_frame = ttk.LabelFrame(
            results_frame, text="Result Details", padding=10
        )
        result_details_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.result_details_text = tk.Text(result_details_frame, wrap="word")
        result_details_scrollbar = ttk.Scrollbar(
            result_details_frame,
            orient="vertical",
            command=self.result_details_text.yview,
        )
        self.result_details_text.configure(yscrollcommand=result_details_scrollbar.set)

        self.result_details_text.pack(side="left", fill="both", expand=True)
        result_details_scrollbar.pack(side="right", fill="y")

        # Bind selection event
        self.results_tree.bind("<<TreeviewSelect>>", self.on_result_select)

    def setup_status_bar(self, parent):
        """Setup the status bar"""
        self.status_bar = ttk.Label(parent, text="Ready", relief="sunken", anchor="w")
        self.status_bar.pack(fill="x", pady=(10, 0))

    def browse_file(self):
        """Browse for Excel file"""
        filename = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if filename:
            self.file_path_var.set(filename)

    def submit_task(self):
        """Submit a new learning task"""
        if not self.background_processor:
            messagebox.showerror("Error", "Background processor not available")
            return

        file_path = self.file_path_var.get().strip()
        if not file_path:
            messagebox.showerror("Error", "Please select an Excel file")
            return

        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Selected file does not exist")
            return

        try:
            # Submit task
            task_id = self.background_processor.submit_learning_task(
                file_path=file_path,
                task_type=self.task_type_var.get(),
                priority=self.priority_var.get(),
                metadata={
                    "submitted_by": "gui",
                    "submitted_at": datetime.now().isoformat(),
                },
            )

            self.submit_status_var.set(
                f"Task submitted successfully! Task ID: {task_id}"
            )
            self.clear_submit_form()

            # Refresh task list
            self.refresh_tasks()

            app_logger.info(f"Submitted learning task: {task_id} for file: {file_path}")

        except Exception as e:
            error_msg = f"Error submitting task: {str(e)}"
            messagebox.showerror("Error", error_msg)
            app_logger.error(error_msg)

    def clear_submit_form(self):
        """Clear the submit form"""
        self.file_path_var.set("")
        self.task_type_var.set("full_pipeline")
        self.priority_var.set(3)
        self.submit_status_var.set("")

    def refresh_tasks(self):
        """Refresh the tasks list"""
        if not self.background_processor:
            return

        # Check if UI components are ready
        if not hasattr(self, "tasks_tree") or not self.tasks_tree:
            return

        try:
            # Clear existing items
            for item in self.tasks_tree.get_children():
                self.tasks_tree.delete(item)

            # Get all tasks
            all_tasks = self.background_processor.get_all_tasks()

            for task_id, task in all_tasks.items():
                # Format progress as percentage
                progress_str = (
                    f"{task.progress:.1f}%" if task.progress is not None else "N/A"
                )

                # Format creation time
                created_str = (
                    task.created_at.strftime("%H:%M:%S") if task.created_at else "N/A"
                )

                # Insert into treeview
                item = self.tasks_tree.insert(
                    "",
                    "end",
                    values=(
                        task_id,
                        os.path.basename(task.file_path),
                        task.task_type,
                        task.status.value,
                        progress_str,
                        created_str,
                    ),
                )

                # Store task reference
                self.task_widgets[item] = task_id

            if hasattr(self, "status_bar") and self.status_bar:
                self.status_bar.config(
                    text=f"Refreshed at {datetime.now().strftime('%H:%M:%S')}"
                )

        except Exception as e:
            app_logger.error(f"Error refreshing tasks: {e}")

    def clear_completed_tasks(self):
        """Clear completed tasks from the display"""
        if not self.background_processor:
            return

        try:
            all_tasks = self.background_processor.get_all_tasks()
            completed_tasks = [
                task_id
                for task_id, task in all_tasks.items()
                if task.status.value in ["completed", "failed", "cancelled"]
            ]

            for task_id in completed_tasks:
                # Remove from treeview
                for item, stored_task_id in list(self.task_widgets.items()):
                    if stored_task_id == task_id:
                        self.tasks_tree.delete(item)
                        del self.task_widgets[item]
                        break

            app_logger.info(f"Cleared {len(completed_tasks)} completed tasks")

        except Exception as e:
            app_logger.error(f"Error clearing completed tasks: {e}")

    def stop_all_tasks(self):
        """Stop all active tasks"""
        if not self.background_processor:
            return

        if messagebox.askyesno(
            "Confirm", "Are you sure you want to stop all active tasks?"
        ):
            try:
                # This would require adding a stop_all_tasks method to BackgroundProcessor
                # For now, we'll just show a message
                messagebox.showinfo(
                    "Info", "Stop all functionality not yet implemented"
                )
                app_logger.info("User requested to stop all tasks")

            except Exception as e:
                app_logger.error(f"Error stopping tasks: {e}")

    def on_task_select(self, event):
        """Handle task selection"""
        selection = self.tasks_tree.selection()
        if not selection:
            return

        item = selection[0]
        task_id = self.task_widgets.get(item)

        if task_id and self.background_processor:
            task = self.background_processor.get_task_status(task_id)
            if task:
                self.show_task_details(task)

    def show_task_details(self, task):
        """Show details for a selected task"""
        progress_str = f"{task.progress:.1f}%" if task.progress is not None else "N/A"
        created_str = (
            task.created_at.strftime("%Y-%m-%d %H:%M:%S") if task.created_at else "N/A"
        )
        started_str = (
            task.started_at.strftime("%Y-%m-%d %H:%M:%S") if task.started_at else "N/A"
        )
        completed_str = (
            task.completed_at.strftime("%Y-%m-%d %H:%M:%S")
            if task.completed_at
            else "N/A"
        )
        metadata_str = json.dumps(task.metadata, indent=2) if task.metadata else "None"
        error_str = task.error if task.error else "None"
        result_summary = (
            self.format_result_summary(task.result) if task.result else "None"
        )

        details = f"""Task Details:
Task ID: {task.task_id}
File: {task.file_path}
Type: {task.task_type}
Status: {task.status.value}
Priority: {task.priority}
Progress: {progress_str}

Created: {created_str}
Started: {started_str}
Completed: {completed_str}

Metadata: {metadata_str}

Error: {error_str}

Result Summary: {result_summary}
"""

        self.task_details_text.delete(1.0, tk.END)
        self.task_details_text.insert(1.0, details)

    def format_result_summary(self, result):
        """Format result summary for display"""
        if not result:
            return "No result available"

        try:
            summary = []
            if "excel_analysis" in result:
                analysis = result["excel_analysis"]
                summary.append(
                    f"Excel Analysis: {len(analysis.get('sheets', []))} sheets analyzed"
                )

            if "formula_learning" in result:
                formulas = result["formula_learning"]
                summary.append(
                    f"Formula Learning: {len(formulas.get('learned_formulas', []))} formulas learned"
                )

            if "chart_learning" in result:
                charts = result["chart_learning"]
                summary.append(
                    f"Chart Learning: {len(charts.get('learned_charts', []))} charts learned"
                )

            return "; ".join(summary) if summary else "Processing completed"

        except Exception as e:
            return f"Error formatting result: {e}"

    def refresh_results(self):
        """Refresh the results list"""
        if not self.background_processor:
            return

        # Check if UI components are ready
        if not hasattr(self, "results_tree") or not self.results_tree:
            return

        try:
            # Clear existing items
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # Get completed tasks
            all_tasks = self.background_processor.get_all_tasks()
            completed_tasks = {
                task_id: task
                for task_id, task in all_tasks.items()
                if task.status.value in ["completed", "failed"]
            }

            for task_id, task in completed_tasks.items():
                # Calculate result size
                result_size = "N/A"
                if task.result:
                    try:
                        result_size = f"{len(str(task.result))} chars"
                    except Exception:
                        result_size = "Unknown"

                # Format completion time
                completed_str = (
                    task.completed_at.strftime("%H:%M:%S")
                    if task.completed_at
                    else "N/A"
                )

                # Insert into treeview
                self.results_tree.insert(
                    "",
                    "end",
                    values=(
                        task_id,
                        os.path.basename(task.file_path),
                        task.task_type,
                        task.status.value,
                        completed_str,
                        result_size,
                    ),
                )

            if hasattr(self, "status_bar") and self.status_bar:
                self.status_bar.config(
                    text=f"Results refreshed at {datetime.now().strftime('%H:%M:%S')}"
                )

        except Exception as e:
            app_logger.error(f"Error refreshing results: {e}")

    def export_results(self):
        """Export results to file"""
        if not self.background_processor:
            return

        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if filename:
            try:
                self.background_processor.export_learning_results(filename)
                messagebox.showinfo("Success", f"Results exported to {filename}")
                app_logger.info(f"Results exported to {filename}")

            except Exception as e:
                error_msg = f"Error exporting results: {str(e)}"
                messagebox.showerror("Error", error_msg)
                app_logger.error(error_msg)

    def clear_results(self):
        """Clear results from display"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        self.result_details_text.delete(1.0, tk.END)
        if hasattr(self, "status_bar") and self.status_bar:
            self.status_bar.config(text="Results cleared")

    def on_result_select(self, event):
        """Handle result selection"""
        selection = self.results_tree.selection()
        if not selection:
            return

        item = selection[0]
        values = self.results_tree.item(item, "values")
        task_id = values[0] if values else None

        if task_id and self.background_processor:
            task = self.background_processor.get_task_status(task_id)
            if task and task.result:
                self.show_result_details(task)

    def show_result_details(self, task):
        """Show details for a selected result"""
        try:
            completed_str = (
                task.completed_at.strftime("%Y-%m-%d %H:%M:%S")
                if task.completed_at
                else "N/A"
            )
            result_str = (
                json.dumps(task.result, indent=2)
                if task.result
                else "No result available"
            )

            details = f"""Result Details for Task: {task.task_id}

File: {task.file_path}
Type: {task.task_type}
Status: {task.status.value}
Completed: {completed_str}

Full Result:
{result_str}
"""

            self.result_details_text.delete(1.0, tk.END)
            self.result_details_text.insert(1.0, details)

        except Exception as e:
            self.result_details_text.delete(1.0, tk.END)
            self.result_details_text.insert(1.0, f"Error displaying result: {e}")

    def on_notification(self, notification):
        """Handle notifications from background processor"""
        try:
            # Update status bar if available
            if hasattr(self, "status_bar") and self.status_bar:
                self.status_bar.config(text=f"Notification: {notification.message}")

            # Show notification in GUI
            if notification.notification_type.value in [
                "task_completed",
                "task_failed",
            ]:
                # Show popup for important notifications
                if notification.notification_type.value == "task_completed":
                    messagebox.showinfo("Task Completed", notification.message)
                else:
                    messagebox.showerror("Task Failed", notification.message)

            # Refresh displays
            self.refresh_tasks()
            self.refresh_results()

            app_logger.info(
                f"Notification received: {notification.notification_type.value} - {notification.message}"
            )

        except Exception as e:
            app_logger.error(f"Error handling notification: {e}")

    def start_ui_updates(self):
        """Start periodic UI updates"""

        def update_loop():
            while not self.stop_updates:
                try:
                    # Update every 2 seconds
                    time.sleep(2)

                    # Refresh displays
                    if self.background_processor:
                        self.refresh_tasks()
                        self.refresh_results()

                except Exception as e:
                    app_logger.error(f"Error in UI update loop: {e}")

        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()

    def stop_ui_updates(self):
        """Stop UI updates"""
        self.stop_updates = True
        if self.update_thread:
            self.update_thread.join(timeout=1)

    def cleanup(self):
        """Cleanup resources"""
        self.stop_ui_updates()

        if self.background_processor:
            try:
                self.background_processor.stop_processing()
            except:
                pass
