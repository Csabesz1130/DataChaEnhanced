import tkinter as tk
from tkinter import ttk, messagebox
from src.utils.logger import app_logger

class SegmentInputFrame(ttk.LabelFrame):
    def __init__(self, parent, callback):
        super().__init__(parent, text="Segment Boundaries", padding="5 5 5 5")
        self.callback = callback
        
        # Default segment values
        self.default_segments = [
            {"start": 35, "end": 234, "is_hyperpol": True},
            {"start": 235, "end": 434, "is_hyperpol": False},
            {"start": 435, "end": 634, "is_hyperpol": True},
            {"start": 635, "end": 834, "is_hyperpol": False}
        ]
        
        self.default_high_segments = {
            'depol_start': 835,
            'depol_end': 1034,
            'hyperpol_start': 1035,
            'hyperpol_end': 1234
        }
        
        # Variables for each segment
        self.segment_vars = []
        for i in range(4):
            segment = {
                'start': tk.StringVar(value=str(self.default_segments[i]['start'])),
                'end': tk.StringVar(value=str(self.default_segments[i]['end'])),
                'is_hyperpol': tk.BooleanVar(value=self.default_segments[i]['is_hyperpol'])
            }
            self.segment_vars.append(segment)
            
        # Use defaults checkbox
        self.use_defaults = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.toggle_inputs()  # Initial state
        
    def setup_ui(self):
        # Use defaults checkbox
        defaults_frame = ttk.Frame(self)
        defaults_frame.pack(fill='x', pady=5)
        ttk.Checkbutton(defaults_frame, text="Use Default Values", 
                       variable=self.use_defaults,
                       command=self.toggle_inputs).pack(pady=5)
                       
        # Analysis segments
        self.analysis_frame = ttk.LabelFrame(self, text="Analysis Segments")
        self.analysis_frame.pack(fill='x', pady=5)

        for i, vars in enumerate(self.segment_vars):
            frame = ttk.Frame(self.analysis_frame)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=f"Segment {i+1}:").pack(side='left', padx=5)
            ttk.Label(frame, text="Start:").pack(side='left')
            ttk.Entry(frame, textvariable=vars['start'], width=8).pack(side='left', padx=2)
            ttk.Label(frame, text="End:").pack(side='left')
            ttk.Entry(frame, textvariable=vars['end'], width=8).pack(side='left', padx=2)
            ttk.Radiobutton(frame, text="Hyperpol", 
                          variable=vars['is_hyperpol'], 
                          value=True).pack(side='left')
            ttk.Radiobutton(frame, text="Depol", 
                          variable=vars['is_hyperpol'], 
                          value=False).pack(side='left')

        # High segment points
        self.high_frame = ttk.LabelFrame(self, text="High Segments")
        self.high_frame.pack(fill='x', pady=5)

        # Depolarization
        depol_frame = ttk.Frame(self.high_frame)
        depol_frame.pack(fill='x', pady=2)
        ttk.Label(depol_frame, text="Depolarization:").pack(side='left', padx=5)
        ttk.Label(depol_frame, text="Start:").pack(side='left')
        self.depol_start = ttk.Entry(depol_frame, width=8)
        self.depol_start.insert(0, str(self.default_high_segments['depol_start']))
        self.depol_start.pack(side='left', padx=2)
        ttk.Label(depol_frame, text="End:").pack(side='left')
        self.depol_end = ttk.Entry(depol_frame, width=8)
        self.depol_end.insert(0, str(self.default_high_segments['depol_end']))
        self.depol_end.pack(side='left', padx=2)

        # Hyperpolarization
        hyperpol_frame = ttk.Frame(self.high_frame)
        hyperpol_frame.pack(fill='x', pady=2)
        ttk.Label(hyperpol_frame, text="Hyperpolarization:").pack(side='left', padx=5)
        ttk.Label(hyperpol_frame, text="Start:").pack(side='left')
        self.hyperpol_start = ttk.Entry(hyperpol_frame, width=8)
        self.hyperpol_start.insert(0, str(self.default_high_segments['hyperpol_start']))
        self.hyperpol_start.pack(side='left', padx=2)
        ttk.Label(hyperpol_frame, text="End:").pack(side='left')
        self.hyperpol_end = ttk.Entry(hyperpol_frame, width=8)
        self.hyperpol_end.insert(0, str(self.default_high_segments['hyperpol_end']))
        self.hyperpol_end.pack(side='left', padx=2)

        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', pady=5)
        ttk.Button(button_frame, text="Apply", 
                  command=self.apply_changes).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset", 
                  command=self.reset_to_defaults).pack(side='left')

    def toggle_inputs(self):
        """Enable/disable input fields based on use_defaults checkbox"""
        state = 'disabled' if self.use_defaults.get() else 'normal'
        
        # Toggle analysis segments
        for vars in self.segment_vars:
            for widget in self.analysis_frame.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for w in widget.winfo_children():
                        if isinstance(w, (ttk.Entry, ttk.Radiobutton)):
                            w.configure(state=state)
        
        # Toggle high segments
        for widget in self.high_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for w in widget.winfo_children():
                    if isinstance(w, ttk.Entry):
                        w.configure(state=state)

    def apply_changes(self):
        """Collect and validate all segment inputs."""
        try:
            if self.use_defaults.get():
                segments = self.default_segments
                high_segments = self.default_high_segments
            else:
                # Get analysis segments
                segments = []
                for vars in self.segment_vars:
                    start = int(vars['start'].get())
                    end = int(vars['end'].get())
                    
                    if start >= end or start < 0:
                        raise ValueError(f"Invalid segment range: {start}-{end}")
                    
                    segments.append({
                        'start': start,
                        'end': end,
                        'is_hyperpol': vars['is_hyperpol'].get()
                    })

                # Get high segments
                high_segments = {
                    'depol_start': int(self.depol_start.get()),
                    'depol_end': int(self.depol_end.get()),
                    'hyperpol_start': int(self.hyperpol_start.get()),
                    'hyperpol_end': int(self.hyperpol_end.get())
                }
                
                # Validate high segments
                if high_segments['depol_end'] <= high_segments['depol_start']:
                    raise ValueError("Invalid depolarization range")
                if high_segments['hyperpol_end'] <= high_segments['hyperpol_start']:
                    raise ValueError("Invalid hyperpolarization range")
                    
            # Call callback with both segment sets
            self.callback((segments, high_segments))

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            
    def reset_to_defaults(self):
        """Reset all values to defaults."""
        self.use_defaults.set(True)
        self.toggle_inputs()
        
        for i, vars in enumerate(self.segment_vars):
            vars['start'].set(str(self.default_segments[i]['start']))
            vars['end'].set(str(self.default_segments[i]['end']))
            vars['is_hyperpol'].set(self.default_segments[i]['is_hyperpol'])
            
        self.depol_start.delete(0, tk.END)
        self.depol_start.insert(0, str(self.default_high_segments['depol_start']))
        self.depol_end.delete(0, tk.END)
        self.depol_end.insert(0, str(self.default_high_segments['depol_end']))
        self.hyperpol_start.delete(0, tk.END)
        self.hyperpol_start.insert(0, str(self.default_high_segments['hyperpol_start']))
        self.hyperpol_end.delete(0, tk.END)
        self.hyperpol_end.insert(0, str(self.default_high_segments['hyperpol_end']))
        
        self.apply_changes()