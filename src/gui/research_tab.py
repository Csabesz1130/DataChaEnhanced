# src/gui/research_tab.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from src.utils.logger import app_logger
from src.research import ModelRegistry, compute_qc_metrics, detect_anomalies, generate_markdown_report


class ResearchTab:
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        self.frame = ttk.LabelFrame(parent, text="Research Mode")
        self.frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.registry = ModelRegistry()
        self._register_default_models()

        self._setup_ui()

    def _setup_ui(self):
        # Gombok
        btns = ttk.Frame(self.frame)
        btns.pack(fill='x', padx=5, pady=5)

        ttk.Button(btns, text="Futtasd QC + Anomália",
                   command=self.run_qc_anomaly).pack(side='left', padx=2)
        ttk.Button(btns, text="Futtasd modelleket",
                   command=self.run_models).pack(side='left', padx=2)
        ttk.Button(btns, text="Export riport (md)",
                   command=self.export_report).pack(side='left', padx=2)

        # Eredmények
        self.result_text = tk.Text(self.frame, height=20, wrap=tk.WORD, font=('Consolas', 9))
        self.result_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Állapot
        self.status = tk.StringVar(value="Kész")
        ttk.Label(self.frame, textvariable=self.status).pack(anchor='w', padx=6)

    def _current_signal(self):
        try:
            proc = getattr(self.main_app, 'action_potential_processor', None)
            if proc and hasattr(proc, 'orange_curve') and proc.orange_curve is not None:
                # használjuk az orange görbét alap jelnek
                t = getattr(proc, 'orange_curve_times', None)
                y = proc.orange_curve
                if t is None:
                    # ha nincs idő, generáljunk minta dt alapján
                    n = len(y)
                    dt = 1e-5
                    t = np.arange(n) * dt
                return {"time": np.asarray(t), "current": np.asarray(y)}
            # fallback: fő jel
            if self.main_app.time_data is not None and self.main_app.data is not None:
                return {"time": np.asarray(self.main_app.time_data), "current": np.asarray(self.main_app.data)}
        except Exception as e:
            app_logger.error(f"Hiba jel olvasásakor: {e}")
        return None

    def _register_default_models(self):
        # példa baseline drift prediktor modell
        def baseline_model(signal):
            t = signal['time']; y = signal['current']
            slope = np.polyfit(t, y, deg=1)[0]
            return {"drift_pa_per_s": float(slope)}

        # példa spike rate modell
        def spike_rate_model(signal, thr=500.0):
            y = signal['current']
            return {"spike_rate_per_s": float(np.mean(y > thr) / (np.median(np.diff(signal['time'])) + 1e-12))}

        self.registry.register("baseline_drift", baseline_model)
        self.registry.register("spike_rate", spike_rate_model)

    def run_qc_anomaly(self):
        sig = self._current_signal()
        if not sig:
            messagebox.showwarning("Nincs jel", "Tölts be adatot vagy futtasd az elemzést.")
            return
        self.status.set("Fut...")
        qc = compute_qc_metrics(sig)
        an = detect_anomalies(sig)
        self._last_qc = qc
        self._last_an = an
        self._write_result_section("QC", qc)
        self._write_result_section("Anomália", an)
        self.status.set("Kész")

    def run_models(self):
        sig = self._current_signal()
        if not sig:
            messagebox.showwarning("Nincs jel", "Tölts be adatot vagy futtasd az elemzést.")
            return
        self.status.set("Modellek futnak...")
        outputs = self.registry.run_all(sig)
        self._last_models = outputs
        self._write_result_section("Modellek", outputs)
        self.status.set("Kész")

    def export_report(self):
        sig = self._current_signal()
        if not sig:
            messagebox.showwarning("Nincs jel", "Tölts be adatot vagy futtasd az elemzést.")
            return

        context = {
            'file': getattr(self.main_app, 'current_file', 'ismeretlen'),
            'qc': getattr(self, '_last_qc', {}),
            'anomaly': getattr(self, '_last_an', {}),
            'models': getattr(self, '_last_models', {}),
        }

        # integrál értékek, ha elérhetők
        try:
            if hasattr(self.main_app, 'action_potential_tab'):
                res = self.main_app.action_potential_tab.get_results_dict()
                if isinstance(res, dict):
                    raw = {
                        'hyperpol_pC': None,
                        'depol_pC': None,
                        'capacitance_nF': None,
                    }
                    # próbáljuk kinyerni a nyers számokat
                    hp = res.get('hyperpol_area'); dp = res.get('depol_area'); cap = res.get('capacitance_nF')
                    def _parse_pc(txt):
                        if isinstance(txt, str):
                            return float(txt.split()[0])
                        return float(txt)
                    try:
                        raw['hyperpol_pC'] = _parse_pc(hp) if hp is not None else None
                        raw['depol_pC'] = _parse_pc(dp) if dp is not None else None
                        raw['capacitance_nF'] = _parse_pc(cap) if cap is not None else None
                    except Exception:
                        pass
                    context['integrals'] = raw
        except Exception:
            pass

        md = generate_markdown_report(context)
        save_path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("All Files", "*.*")]
        )
        if not save_path:
            return
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(md)
            messagebox.showinfo("Export", f"Riport elmentve: {save_path}")
        except Exception as e:
            messagebox.showerror("Hiba", str(e))

    def _write_result_section(self, title, obj):
        self.result_text.config(state='normal')
        self.result_text.insert('end', f"\n=== {title} ===\n")
        try:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    self.result_text.insert('end', f"{k}: {v}\n")
            else:
                self.result_text.insert('end', str(obj) + "\n")
        except Exception as e:
            self.result_text.insert('end', f"[hiba] {e}\n")
        self.result_text.config(state='disabled')