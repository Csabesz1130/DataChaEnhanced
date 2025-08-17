#!/usr/bin/env python3
"""
AI Excel Learning Demo Tab

Ez a modul egy dedikált GUI tab-ot biztosít a kész tanpéldákhoz és 
az AI Excel Learning dashboard integrációjához. A felhasználók itt
megismerhetik a rendszer képességeit és interaktívan tesztelhetik
a különböző funkciókat.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import webbrowser
import subprocess
import sys

from src.utils.logger import app_logger

class AIExcelLearningDemoTab:
    """Dedikált tab a kész tanpéldákhoz és AI Excel Learning dashboard-hoz"""
    
    def __init__(self, notebook, app):
        """Inicializálja az AI Excel Learning Demo tab-ot"""
        self.notebook = notebook
        self.app = app
        self.frame = ttk.Frame(notebook)
        
        # Demo állapot
        self.demo_running = False
        self.current_demo = None
        self.demo_results = {}
        
        # AI Dashboard integráció
        self.dashboard_process = None
        self.dashboard_url = "http://localhost:8501"
        
        # Setup UI
        self.setup_ui()
        
        app_logger.info("AI Excel Learning Demo tab inicializálva")
    
    def setup_ui(self):
        """Beállítja a felhasználói felületet"""
        # Fő cím
        title_frame = ttk.Frame(self.frame)
        title_frame.pack(fill='x', padx=10, pady=10)
        
        title_label = ttk.Label(
            title_frame, 
            text="🤖 AI Excel Learning - Kész Tanpéldák & Dashboard", 
            font=('Arial', 16, 'bold')
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Ismerd meg a rendszer képességeit és teszteld interaktívan",
            font=('Arial', 10)
        )
        subtitle_label.pack()
        
        # Fő konténer
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Bal oldali panel - Tanpéldák
        left_panel = ttk.LabelFrame(main_container, text="📚 Kész Tanpéldák", padding=10)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.setup_examples_panel(left_panel)
        
        # Jobb oldali panel - Dashboard & Integráció
        right_panel = ttk.LabelFrame(main_container, text="📊 AI Dashboard & Integráció", padding=10)
        right_panel.panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.setup_dashboard_panel(right_panel)
        
        # Alsó panel - Eredmények & Logok
        bottom_panel = ttk.LabelFrame(self.frame, text="📋 Eredmények & Logok", padding=10)
        bottom_panel.pack(fill='x', padx=10, pady=5)
        
        self.setup_results_panel(bottom_panel)
    
    def setup_examples_panel(self, parent):
        """Beállítja a tanpéldák panelt"""
        # Tanpélda kategóriák
        categories_frame = ttk.Frame(parent)
        categories_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(categories_frame, text="Válassz kategóriát:").pack(anchor='w')
        
        self.category_var = tk.StringVar(value="basic")
        category_combo = ttk.Combobox(
            categories_frame,
            textvariable=self.category_var,
            values=[
                ("basic", "Alapvető Excel elemzés"),
                ("charts", "Grafikonok tanítása"),
                ("formulas", "Képletek tanítása"),
                ("ml_models", "ML modellek"),
                ("pipeline", "Teljes tanulási folyamat"),
                ("advanced", "Haladó funkciók")
            ],
            state="readonly",
            width=30
        )
        category_combo.pack(anchor='w', pady=5)
        category_combo.bind('<<ComboboxSelected>>', self.on_category_changed)
        
        # Tanpélda leírás
        desc_frame = ttk.Frame(parent)
        desc_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(desc_frame, text="Tanpélda leírása:").pack(anchor='w')
        
        self.description_text = scrolledtext.ScrolledText(
            desc_frame, 
            height=6, 
            wrap=tk.WORD,
            state='disabled'
        )
        self.description_text.pack(fill='x', pady=5)
        
        # Tanpélda futtatása
        demo_frame = ttk.Frame(parent)
        demo_frame.pack(fill='x', pady=(0, 10))
        
        self.run_demo_btn = ttk.Button(
            demo_frame,
            text="▶ Futtatás",
            command=self.run_selected_demo,
            style='Accent.TButton'
        )
        self.run_demo_btn.pack(side='left', padx=(0, 10))
        
        self.stop_demo_btn = ttk.Button(
            demo_frame,
            text="⏹ Leállítás",
            command=self.stop_demo,
            state='disabled'
        )
        self.stop_demo_btn.pack(side='left')
        
        # Folyamat állapot
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(progress_frame, text="Folyamat állapota:").pack(anchor='w')
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='indeterminate'
        )
        self.progress_bar.pack(fill='x', pady=5)
        
        self.status_label = ttk.Label(
            progress_frame,
            text="Kész a futtatásra",
            font=('Arial', 9)
        )
        self.status_label.pack(anchor='w')
        
        # Kezdeti leírás betöltése
        self.update_description()
    
    def setup_dashboard_panel(self, parent):
        """Beállítja a dashboard panelt"""
        # Dashboard indítása
        dashboard_start_frame = ttk.Frame(parent)
        dashboard_start_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(
            dashboard_start_frame, 
            text="AI Monitoring Dashboard indítása:"
        ).pack(anchor='w')
        
        self.start_dashboard_btn = ttk.Button(
            dashboard_start_frame,
            text="🚀 Indítás",
            command=self.start_dashboard,
            style='Accent.TButton'
        )
        self.start_dashboard_btn.pack(side='left', padx=(0, 10))
        
        self.stop_dashboard_btn = ttk.Button(
            dashboard_start_frame,
            text="⏹ Leállítás",
            command=self.stop_dashboard,
            state='disabled'
            )
        self.stop_dashboard_btn.pack(side='left')
        
        # Dashboard állapot
        dashboard_status_frame = ttk.Frame(parent)
        dashboard_status_frame.pack(fill='x', pady=(0, 10))
        
        self.dashboard_status_label = ttk.Label(
            dashboard_status_frame,
            text="Dashboard nincs elindítva",
            font=('Arial', 9)
        )
        self.dashboard_status_label.pack(anchor='w')
        
        # Dashboard megnyitása
        open_frame = ttk.Frame(parent)
        open_frame.pack(fill='x', pady=(0, 10))
        
        self.open_dashboard_btn = ttk.Button(
            open_frame,
            text="🌐 Megnyitás böngészőben",
            command=self.open_dashboard_in_browser,
            state='disabled'
        )
        self.open_dashboard_btn.pack(side='left')
        
        # Integrációs opciók
        integration_frame = ttk.LabelFrame(parent, text="🔗 Integrációs Opciók", padding=5)
        integration_frame.pack(fill='x', pady=(0, 10))
        
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_refresh_check = ttk.Checkbutton(
            integration_frame,
            text="Automatikus frissítés",
            variable=self.auto_refresh_var
        )
        auto_refresh_check.pack(anchor='w')
        
        self.real_time_var = tk.BooleanVar(value=True)
        real_time_check = ttk.Checkbutton(
            integration_frame,
            text="Valós idejű monitoring",
            variable=self.real_time_var
        )
        real_time_check.pack(anchor='w')
        
        # Export opciók
        export_frame = ttk.LabelFrame(parent, text="📤 Export Opciók", padding=5)
        export_frame.pack(fill='x')
        
        export_btn = ttk.Button(
            export_frame,
            text="📊 Eredmények exportálása",
            command=self.export_results
        )
        export_btn.pack(side='left', padx=(0, 10))
        
        report_btn = ttk.Button(
            export_frame,
            text="📋 Jelentés generálása",
            command=self.generate_report
        )
        report_btn.pack(side='left')
    
    def setup_results_panel(self, parent):
        """Beállítja az eredmények panelt"""
        # Eredmények megjelenítése
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill='both', expand=True)
        
        # Eredmények listája
        list_frame = ttk.Frame(results_frame)
        list_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        ttk.Label(list_frame, text="Futtatott tanpéldák:").pack(anchor='w')
        
        self.results_tree = ttk.Treeview(
            list_frame,
            columns=('Dátum', 'Kategória', 'Állapot', 'Időtartam'),
            show='headings',
            height=6
        )
        
        self.results_tree.heading('Dátum', text='Dátum')
        self.results_tree.heading('Kategória', text='Kategória')
        self.results_tree.heading('Állapot', text='Állapot')
        self.results_tree.heading('Időtartam', text='Időtartam')
        
        self.results_tree.column('Dátum', width=120)
        self.results_tree.column('Kategória', width=150)
        self.results_tree.column('Állapot', width=100)
        self.results_tree.column('Időtartam', width=100)
        
        self.results_tree.pack(fill='both', expand=True)
        
        # Részletek
        details_frame = ttk.Frame(results_frame)
        details_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        ttk.Label(details_frame, text="Részletek:").pack(anchor='w')
        
        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            height=6,
            wrap=tk.WORD,
            state='disabled'
        )
        self.details_text.pack(fill='both', expand=True)
        
        # Eredmények betöltése
        self.load_results()
    
    def on_category_changed(self, event=None):
        """Kategória változásakor frissíti a leírást"""
        self.update_description()
    
    def update_description(self):
        """Frissíti a kiválasztott kategória leírását"""
        category = self.category_var.get()
        descriptions = {
            "basic": """Alapvető Excel elemzés:
• Excel fájlok struktúrájának felismerése
• Adatok típusának és formátumának elemzése
• Oszlopok és sorok kapcsolatainak felderítése
• Egyszerű statisztikák generálása""",
            
            "charts": """Grafikonok tanítása:
• Excel grafikonok típusának felismerése
• Adatok és grafikonok közötti kapcsolatok
• Grafikon stílusok és formázások tanítása
• Új grafikonok generálása a tanult minták alapján""",
            
            "formulas": """Képletek tanítása:
• Excel képletek mintázatainak felismerése
• Matematikai és logikai kapcsolatok tanítása
• Képlet függőségek és referenciák elemzése
• Új képletek generálása a tanult minták alapján""",
            
            "ml_models": """ML modellek:
• Gépi tanulási modellek Excel adatokra
• Prediktív elemzés és trend felismerés
• Anomália detektálás
• Modellek teljesítményének monitorozása""",
            
            "pipeline": """Teljes tanulási folyamat:
• End-to-end Excel tanulási folyamat
• Automatikus adatfeldolgozás és elemzés
• Folyamatos tanulás és fejlesztés
• Teljesítmény optimalizálás""",
            
            "advanced": """Haladó funkciók:
• Komplex Excel munkafüzetek elemzése
• Több munkalap közötti kapcsolatok
• Makrók és VBA kód elemzése
• Automatikus dokumentáció generálás"""
        }
        
        description = descriptions.get(category, "Kategória leírása nem elérhető")
        
        self.description_text.config(state='normal')
        self.description_text.delete(1.0, tk.END)
        self.description_text.insert(1.0, description)
        self.description_text.config(state='disabled')
    
    def run_selected_demo(self):
        """Futtatja a kiválasztott tanpéldát"""
        if self.demo_running:
            messagebox.showwarning("Figyelmeztetés", "Már fut egy tanpélda!")
            return
        
        category = self.category_var.get()
        if not category:
            messagebox.showerror("Hiba", "Válassz ki egy kategóriát!")
            return
        
        # UI frissítése
        self.demo_running = True
        self.run_demo_btn.config(state='disabled')
        self.stop_demo_btn.config(state='normal')
        self.progress_bar.start()
        self.status_label.config(text="Tanpélda futtatása...")
        
        # Demo futtatása külön szálban
        self.demo_thread = threading.Thread(
            target=self._run_demo_worker,
            args=(category,)
        )
        self.demo_thread.daemon = True
        self.demo_thread.start()
    
    def _run_demo_worker(self, category):
        """Worker szál a demo futtatásához"""
        try:
            start_time = time.time()
            
            # Demo futtatása a kiválasztott kategóriához
            if category == "basic":
                result = self._run_basic_demo()
            elif category == "charts":
                result = self._run_charts_demo()
            elif category == "formulas":
                result = self._run_formulas_demo()
            elif category == "ml_models":
                result = self._run_ml_models_demo()
            elif category == "pipeline":
                result = self._run_pipeline_demo()
            elif category == "advanced":
                result = self._run_advanced_demo()
            else:
                result = {"success": False, "error": "Ismeretlen kategória"}
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Eredmény mentése
            result_data = {
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "duration": f"{duration:.2f}s",
                "success": result.get("success", False),
                "details": result
            }
            
            self.demo_results[category] = result_data
            
            # UI frissítése fő szálban
            self.frame.after(0, self._demo_completed, result_data)
            
        except Exception as e:
            error_result = {
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "duration": "0s",
                "success": False,
                "details": {"error": str(e)}
            }
            self.frame.after(0, self._demo_completed, error_result)
    
    def _run_basic_demo(self):
        """Alapvető Excel elemzés demo"""
        try:
            # Itt hívnánk meg a valós AI Excel Learning funkciókat
            # Példa implementáció
            time.sleep(2)  # Szimulált feldolgozás
            
            return {
                "success": True,
                "message": "Alapvető Excel elemzés sikeresen lefutott",
                "files_analyzed": 3,
                "patterns_found": 5,
                "processing_time": "2.1s"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_charts_demo(self):
        """Grafikonok tanítása demo"""
        try:
            time.sleep(3)  # Szimulált feldolgozás
            
            return {
                "success": True,
                "message": "Grafikon tanítás sikeresen lefutott",
                "charts_analyzed": 8,
                "chart_types_learned": ["scatter", "line", "bar", "pie"],
                "processing_time": "3.2s"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_formulas_demo(self):
        """Képletek tanítása demo"""
        try:
            time.sleep(2.5)  # Szimulált feldolgozás
            
            return {
                "success": True,
                "message": "Képlet tanítás sikeresen lefutott",
                "formulas_analyzed": 12,
                "patterns_learned": ["mathematical", "logical", "lookup"],
                "processing_time": "2.5s"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_ml_models_demo(self):
        """ML modellek demo"""
        try:
            time.sleep(4)  # Szimulált feldolgozás
            
            return {
                "success": True,
                "message": "ML modellek sikeresen tanultak",
                "models_trained": 2,
                "accuracy": "94.2%",
                "processing_time": "4.1s"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_pipeline_demo(self):
        """Teljes tanulási folyamat demo"""
        try:
            time.sleep(5)  # Szimulált feldolgozás
            
            return {
                "success": True,
                "message": "Teljes tanulási folyamat sikeresen lefutott",
                "total_files": 15,
                "total_patterns": 28,
                "processing_time": "5.3s"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_advanced_demo(self):
        """Haladó funkciók demo"""
        try:
            time.sleep(3.5)  # Szimulált feldolgozás
            
            return {
                "success": True,
                "message": "Haladó funkciók sikeresen lefutottak",
                "complex_workbooks": 4,
                "vba_analysis": True,
                "processing_time": "3.5s"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _demo_completed(self, result_data):
        """Demo befejezése után frissíti a UI-t"""
        self.demo_running = False
        self.run_demo_btn.config(state='normal')
        self.stop_demo_btn.config(state='disabled')
        self.progress_bar.stop()
        
        if result_data["success"]:
            self.status_label.config(text="Tanpélda sikeresen lefutott")
            messagebox.showinfo("Siker", f"Tanpélda sikeresen lefutott!\nIdőtartam: {result_data['duration']}")
        else:
            self.status_label.config(text="Tanpélda hibával lefutott")
            messagebox.showerror("Hiba", f"Tanpélda hibával lefutott:\n{result_data['details'].get('error', 'Ismeretlen hiba')}")
        
        # Eredmények frissítése
        self.load_results()
    
    def stop_demo(self):
        """Leállítja a futó tanpéldát"""
        if self.demo_running:
            self.demo_running = False
            self.status_label.config(text="Tanpélda leállítva")
            self.progress_bar.stop()
            self.run_demo_btn.config(state='normal')
            self.stop_demo_btn.config(state='disabled')
    
    def start_dashboard(self):
        """Indítja az AI Dashboard-ot"""
        try:
            # Streamlit dashboard indítása
            dashboard_script = os.path.join(
                os.path.dirname(__file__), 
                "..", "ai_excel_learning", "ai_dashboard.py"
            )
            
            if not os.path.exists(dashboard_script):
                messagebox.showerror("Hiba", "Dashboard script nem található!")
                return
            
            # Dashboard indítása külön folyamatban
            self.dashboard_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", dashboard_script,
                "--server.port", "8501",
                "--server.headless", "true"
            ])
            
            # Várakozás a dashboard elindulására
            time.sleep(3)
            
            # UI frissítése
            self.start_dashboard_btn.config(state='disabled')
            self.stop_dashboard_btn.config(state='normal')
            self.open_dashboard_btn.config(state='normal')
            self.dashboard_status_label.config(
                text="Dashboard fut (http://localhost:8501)",
                foreground="green"
            )
            
            messagebox.showinfo("Siker", "AI Dashboard sikeresen elindult!")
            
        except Exception as e:
            messagebox.showerror("Hiba", f"Dashboard indítása sikertelen:\n{str(e)}")
    
    def stop_dashboard(self):
        """Leállítja az AI Dashboard-ot"""
        if self.dashboard_process:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
            
            self.dashboard_process = None
            
            # UI frissítése
            self.start_dashboard_btn.config(state='normal')
            self.stop_dashboard_btn.config(state='disabled')
            self.open_dashboard_btn.config(state='disabled')
            self.dashboard_status_label.config(
                text="Dashboard nincs elindítva",
                foreground="black"
            )
    
    def open_dashboard_in_browser(self):
        """Megnyitja a dashboard-ot a böngészőben"""
        try:
            webbrowser.open(self.dashboard_url)
        except Exception as e:
            messagebox.showerror("Hiba", f"Böngésző megnyitása sikertelen:\n{str(e)}")
    
    def export_results(self):
        """Exportálja az eredményeket"""
        if not self.demo_results:
            messagebox.showinfo("Információ", "Nincsenek exportálható eredmények!")
            return
        
        try:
            filename = f"ai_excel_learning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename
            )
            
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.demo_results, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Siker", f"Eredmények exportálva: {filepath}")
        
        except Exception as e:
            messagebox.showerror("Hiba", f"Exportálás sikertelen:\n{str(e)}")
    
    def generate_report(self):
        """Generál egy összefoglaló jelentést"""
        if not self.demo_results:
            messagebox.showinfo("Információ", "Nincs adat a jelentés generálásához!")
            return
        
        try:
            # Egyszerű jelentés generálása
            report = self._create_report()
            
            filename = f"ai_excel_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialname=filename
            )
            
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                messagebox.showinfo("Siker", f"Jelentés generálva: {filepath}")
        
        except Exception as e:
            messagebox.showerror("Hiba", f"Jelentés generálása sikertelen:\n{str(e)}")
    
    def _create_report(self):
        """Létrehozza az összefoglaló jelentést"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AI EXCEL LEARNING - ÖSSZEFOGLALÓ JELENTÉS")
        report_lines.append("=" * 60)
        report_lines.append(f"Generálva: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Összesítés
        total_demos = len(self.demo_results)
        successful_demos = sum(1 for r in self.demo_results.values() if r["success"])
        failed_demos = total_demos - successful_demos
        
        report_lines.append("ÖSSZESÍTÉS:")
        report_lines.append(f"  - Összes tanpélda: {total_demos}")
        report_lines.append(f"  - Sikeres: {successful_demos}")
        report_lines.append(f"  - Sikertelen: {failed_demos}")
        report_lines.append(f"  - Sikerességi arány: {(successful_demos/total_demos*100):.1f}%")
        report_lines.append("")
        
        # Részletes eredmények
        report_lines.append("RÉSZLETES EREDMÉNYEK:")
        report_lines.append("-" * 40)
        
        for category, result in self.demo_results.items():
            report_lines.append(f"Kategória: {category}")
            report_lines.append(f"  Dátum: {result['timestamp']}")
            report_lines.append(f"  Időtartam: {result['duration']}")
            report_lines.append(f"  Állapot: {'Sikeres' if result['success'] else 'Sikertelen'}")
            
            if result['success'] and 'details' in result:
                details = result['details']
                if 'message' in details:
                    report_lines.append(f"  Üzenet: {details['message']}")
                if 'files_analyzed' in details:
                    report_lines.append(f"  Elemzett fájlok: {details['files_analyzed']}")
                if 'patterns_found' in details:
                    report_lines.append(f"  Talált mintázatok: {details['patterns_found']}")
            
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def load_results(self):
        """Betölti a mentett eredményeket"""
        # Eredmények törlése
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Eredmények hozzáadása
        for category, result in self.demo_results.items():
            status = "✅ Sikeres" if result["success"] else "❌ Sikertelen"
            self.results_tree.insert("", "end", values=(
                result["timestamp"][:19].replace("T", " "),
                category,
                status,
                result["duration"]
            ))
        
        # Eredmények kiválasztásának kezelése
        self.results_tree.bind('<<TreeviewSelect>>', self.on_result_selected)
    
    def on_result_selected(self, event=None):
        """Eredmény kiválasztásakor frissíti a részleteket"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = self.results_tree.item(selection[0])
        category = item['values'][1]
        
        if category in self.demo_results:
            result = self.demo_results[category]
            details = self._format_result_details(result)
            
            self.details_text.config(state='normal')
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, details)
            self.details_text.config(state='disabled')
    
    def _format_result_details(self, result):
        """Formázza az eredmény részleteit"""
        lines = []
        lines.append(f"Kategória: {result['category']}")
        lines.append(f"Dátum: {result['timestamp']}")
        lines.append(f"Időtartam: {result['duration']}")
        lines.append(f"Állapot: {'Sikeres' if result['success'] else 'Sikertelen'}")
        lines.append("")
        
        if 'details' in result:
            details = result['details']
            if result['success']:
                lines.append("Részletek:")
                for key, value in details.items():
                    if key != 'success':
                        lines.append(f"  {key}: {value}")
            else:
                lines.append(f"Hiba: {details.get('error', 'Ismeretlen hiba')}")
        
        return "\n".join(lines)
    
    def cleanup(self):
        """Tisztítja fel az erőforrásokat"""
        if self.dashboard_process:
            self.stop_dashboard()
        
        if hasattr(self, 'demo_thread') and self.demo_thread.is_alive():
            self.demo_running = False
            self.demo_thread.join(timeout=1) 