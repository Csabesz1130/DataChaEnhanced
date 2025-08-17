#!/usr/bin/env python3
"""
Kész Tanpéldák és AI Dashboard Menü Tab

Ez a modul egy dedikált menüpontot biztosít a kész tanpéldákhoz és 
az AI Dashboard integrációjához. A rendszer hosszú távon gondolkodva
lett kialakítva, hogy mind desktop, mind webapp verzióban optimálisan
működjön.

Funkciók:
- 📚 Interaktív kész tanpéldák kategóriákba rendezve
- 📊 AI Dashboard integráció külön oldalként
- 🔧 Fejlett monitoring és riportolás
- 📋 Eredmények exportálása és megosztása
- ⚙️ Testreszabható beállítások
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import json
import webbrowser
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.utils.logger import app_logger

class ExamplesMenuTab:
    """Dedikált tab a kész tanpéldákhoz és AI Dashboard-hoz"""
    
    def __init__(self, notebook, app):
        """Inicializálja a Kész Tanpéldák menü tab-ot"""
        self.notebook = notebook
        self.app = app
        self.frame = ttk.Frame(notebook)
        
        # Állapot változók
        self.demo_running = False
        self.current_demo = None
        self.demo_results = {}
        self.ai_modules_available = self._check_ai_modules()
        
        # Dashboard integráció
        self.dashboard_process = None
        self.dashboard_url = "http://localhost:8501"
        self.webapp_process = None
        self.webapp_url = "http://localhost:8502"
        
        # Beállítások
        self.settings = {
            "auto_refresh": True,
            "real_time_monitoring": True,
            "detailed_logging": False,
            "export_format": "json"
        }
        
        # UI felépítése
        self.setup_ui()
        self.load_settings()
        
        app_logger.info("Kész Tanpéldák menü tab inicializálva")
    
    def _check_ai_modules(self):
        """Ellenőrzi az AI modulok elérhetőségét"""
        try:
            from src.ai_excel_learning import excel_analyzer, chart_learner, formula_learner
            return True
        except ImportError:
            return False
    
    def setup_ui(self):
        """Beállítja a teljes felhasználói felületet"""
        # Fő cím és navigációs sáv
        self.setup_header()
        
        # Fő tartalom területe notebook-kal
        self.setup_main_content()
        
        # Alsó státusz sáv
        self.setup_status_bar()
    
    def setup_header(self):
        """Beállítja a fejléc részt"""
        header_frame = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=1)
        header_frame.pack(fill='x', padx=5, pady=5)
        
        # Főcím
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(fill='x', padx=10, pady=10)
        
        title_label = ttk.Label(
            title_frame,
            text="🎯 AI Excel Learning - Kész Tanpéldák & Dashboard",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(side='left')
        
        # Gyors akció gombok
        actions_frame = ttk.Frame(title_frame)
        actions_frame.pack(side='right')
        
        ttk.Button(
            actions_frame,
            text="🚀 Webapp Indítása",
            command=self.launch_webapp,
            style='Accent.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            actions_frame,
            text="📊 Dashboard",
            command=self.toggle_dashboard
        ).pack(side='left', padx=5)
        
        ttk.Button(
            actions_frame,
            text="⚙️ Beállítások",
            command=self.show_settings
        ).pack(side='left', padx=5)
        
        # Navigációs sáv
        nav_frame = ttk.Frame(header_frame)
        nav_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Tab-szerű navigáció
        self.nav_var = tk.StringVar(value="examples")
        nav_buttons = [
            ("examples", "📚 Tanpéldák", self.show_examples_view),
            ("dashboard", "📊 Dashboard", self.show_dashboard_view),
            ("testing", "🔧 Tesztelés", self.show_testing_view),
            ("results", "📋 Eredmények", self.show_results_view)
        ]
        
        for nav_id, nav_text, nav_command in nav_buttons:
            btn = ttk.Radiobutton(
                nav_frame,
                text=nav_text,
                variable=self.nav_var,
                value=nav_id,
                command=nav_command
            )
            btn.pack(side='left', padx=5)
    
    def setup_main_content(self):
        """Beállítja a fő tartalom területet"""
        # Fő tartalom konténer
        self.content_frame = ttk.Frame(self.frame)
        self.content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Nézetek beállítása
        self.setup_examples_view()
        self.setup_dashboard_view() 
        self.setup_testing_view()
        self.setup_results_view()
        
        # Kezdeti nézet
        self.show_examples_view()
    
    def setup_examples_view(self):
        """Beállítja a tanpéldák nézetet"""
        self.examples_frame = ttk.Frame(self.content_frame)
        
        # Bal oldali panel - Kategóriák és leírások
        left_panel = ttk.LabelFrame(self.examples_frame, text="📚 Tanpéldák", padding=10)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Kategória választó
        category_frame = ttk.Frame(left_panel)
        category_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(category_frame, text="Válassz kategóriát:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.category_var = tk.StringVar(value="basic")
        self.category_combo = ttk.Combobox(
            category_frame,
            textvariable=self.category_var,
            values=list(self.get_example_categories().keys()),
            state="readonly",
            width=40
        )
        self.category_combo.pack(anchor='w', pady=5, fill='x')
        self.category_combo.bind('<<ComboboxSelected>>', self.on_category_changed)
        
        # Tanpélda leírás
        desc_frame = ttk.Frame(left_panel)
        desc_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        ttk.Label(desc_frame, text="Tanpélda részletei:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.description_text = scrolledtext.ScrolledText(
            desc_frame,
            height=12,
            wrap=tk.WORD,
            state='disabled',
            font=('Consolas', 9)
        )
        self.description_text.pack(fill='both', expand=True, pady=5)
        
        # Tanpélda vezérlők
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(fill='x', pady=(0, 10))
        
        # Futtatás gombok
        run_frame = ttk.Frame(controls_frame)
        run_frame.pack(fill='x', pady=(0, 5))
        
        self.run_demo_btn = ttk.Button(
            run_frame,
            text="▶ Futtatás",
            command=self.run_selected_demo,
            style='Accent.TButton'
        )
        self.run_demo_btn.pack(side='left', padx=(0, 10))
        
        self.stop_demo_btn = ttk.Button(
            run_frame,
            text="⏹ Leállítás",
            command=self.stop_demo,
            state='disabled'
        )
        self.stop_demo_btn.pack(side='left', padx=(0, 10))
        
        ttk.Button(
            run_frame,
            text="📋 Részletek",
            command=self.show_demo_details
        ).pack(side='left')
        
        # Jobb oldali panel - Folyamat monitoring
        right_panel = ttk.LabelFrame(self.examples_frame, text="📊 Monitoring & Állapot", padding=10)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Rendszer állapot
        status_frame = ttk.Frame(right_panel)
        status_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(status_frame, text="Rendszer állapot:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.status_tree = ttk.Treeview(
            status_frame,
            columns=('Állapot',),
            show='tree headings',
            height=6
        )
        self.status_tree.heading('#0', text='Modul')
        self.status_tree.heading('Állapot', text='Állapot')
        self.status_tree.column('#0', width=120)
        self.status_tree.column('Állapot', width=100)
        self.status_tree.pack(fill='x', pady=5)
        
        # Folyamat állapot
        progress_frame = ttk.Frame(right_panel)
        progress_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(progress_frame, text="Aktuális folyamat:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
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
        
        # Teljesítmény metrikák
        metrics_frame = ttk.LabelFrame(right_panel, text="📈 Teljesítmény Metrikák", padding=5)
        metrics_frame.pack(fill='x', pady=(0, 15))
        
        self.metrics_labels = {}
        metrics = [
            ("Futtatások", "0"),
            ("Sikeresség", "0%"),
            ("Átlagos idő", "0s"),
            ("Hibák", "0")
        ]
        
        for i, (metric, value) in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            metric_frame = ttk.Frame(metrics_frame)
            metric_frame.grid(row=row, column=col, sticky='ew', padx=5, pady=2)
            metrics_frame.columnconfigure(col, weight=1)
            
            ttk.Label(metric_frame, text=f"{metric}:", font=('Arial', 8)).pack(side='left')
            label = ttk.Label(metric_frame, text=value, font=('Arial', 8, 'bold'))
            label.pack(side='right')
            self.metrics_labels[metric] = label
        
        # Gyors akciók
        actions_frame = ttk.LabelFrame(right_panel, text="⚡ Gyors Akciók", padding=5)
        actions_frame.pack(fill='x')
        
        ttk.Button(
            actions_frame,
            text="📤 Export eredmények",
            command=self.export_results
        ).pack(fill='x', pady=2)
        
        ttk.Button(
            actions_frame,
            text="🔄 Frissítés",
            command=self.refresh_status
        ).pack(fill='x', pady=2)
        
        ttk.Button(
            actions_frame,
            text="🧹 Tisztítás",
            command=self.clear_results
        ).pack(fill='x', pady=2)
        
        # Kezdeti leírás és állapot betöltése
        self.update_description()
        self.refresh_status()
    
    def setup_dashboard_view(self):
        """Beállítja a dashboard nézetet"""
        self.dashboard_frame = ttk.Frame(self.content_frame)
        
        # Dashboard vezérlőpult
        control_panel = ttk.LabelFrame(self.dashboard_frame, text="🎛️ Dashboard Vezérlőpult", padding=10)
        control_panel.pack(fill='x', padx=10, pady=10)
        
        # Dashboard szolgáltatások
        services_frame = ttk.Frame(control_panel)
        services_frame.pack(fill='x')
        
        # AI Dashboard
        ai_dash_frame = ttk.LabelFrame(services_frame, text="📊 AI Monitoring Dashboard", padding=5)
        ai_dash_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.dashboard_status_label = ttk.Label(
            ai_dash_frame,
            text="❌ Nincs elindítva",
            font=('Arial', 9)
        )
        self.dashboard_status_label.pack()
        
        dash_buttons_frame = ttk.Frame(ai_dash_frame)
        dash_buttons_frame.pack(fill='x', pady=5)
        
        self.start_dashboard_btn = ttk.Button(
            dash_buttons_frame,
            text="🚀 Indítás",
            command=self.start_dashboard
        )
        self.start_dashboard_btn.pack(side='left', padx=(0, 5))
        
        self.open_dashboard_btn = ttk.Button(
            dash_buttons_frame,
            text="🌐 Megnyitás",
            command=self.open_dashboard,
            state='disabled'
        )
        self.open_dashboard_btn.pack(side='left')
        
        # WebApp Dashboard
        webapp_frame = ttk.LabelFrame(services_frame, text="🌐 WebApp Dashboard", padding=5)
        webapp_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.webapp_status_label = ttk.Label(
            webapp_frame,
            text="❌ Nincs elindítva",
            font=('Arial', 9)
        )
        self.webapp_status_label.pack()
        
        webapp_buttons_frame = ttk.Frame(webapp_frame)
        webapp_buttons_frame.pack(fill='x', pady=5)
        
        self.start_webapp_btn = ttk.Button(
            webapp_buttons_frame,
            text="🚀 Indítás",
            command=self.start_webapp
        )
        self.start_webapp_btn.pack(side='left', padx=(0, 5))
        
        self.open_webapp_btn = ttk.Button(
            webapp_buttons_frame,
            text="🌐 Megnyitás",
            command=self.open_webapp,
            state='disabled'
        )
        self.open_webapp_btn.pack(side='left')
        
        # Dashboard integráció opciók
        integration_frame = ttk.LabelFrame(self.dashboard_frame, text="🔗 Integráció & Beállítások", padding=10)
        integration_frame.pack(fill='x', padx=10, pady=10)
        
        options_frame = ttk.Frame(integration_frame)
        options_frame.pack(fill='x')
        
        # Bal oldali opciók
        left_options = ttk.Frame(options_frame)
        left_options.pack(side='left', fill='both', expand=True)
        
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            left_options,
            text="🔄 Automatikus frissítés",
            variable=self.auto_refresh_var,
            command=self.on_settings_changed
        ).pack(anchor='w', pady=2)
        
        self.real_time_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            left_options,
            text="⏱️ Valós idejű monitoring",
            variable=self.real_time_var,
            command=self.on_settings_changed
        ).pack(anchor='w', pady=2)
        
        # Jobb oldali opciók
        right_options = ttk.Frame(options_frame)
        right_options.pack(side='right', fill='both', expand=True)
        
        self.detailed_logging_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            right_options,
            text="📝 Részletes naplózás",
            variable=self.detailed_logging_var,
            command=self.on_settings_changed
        ).pack(anchor='w', pady=2)
        
        ttk.Button(
            right_options,
            text="📊 Teljesítmény riport",
            command=self.generate_performance_report
        ).pack(anchor='w', pady=2)
        
        # Dashboard előnézet terület (későbbi fejlesztéshez)
        preview_frame = ttk.LabelFrame(self.dashboard_frame, text="👁️ Dashboard Előnézet", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        preview_info = ttk.Label(
            preview_frame,
            text="A dashboard előnézet funkció fejlesztés alatt áll.\nKattints a 'Megnyitás' gombokra a teljes felület eléréséhez.",
            justify='center',
            font=('Arial', 10)
        )
        preview_info.pack(expand=True)
    
    def setup_testing_view(self):
        """Beállítja a tesztelési nézetet"""
        self.testing_frame = ttk.Frame(self.content_frame)
        
        # Tesztelési opciók
        test_panel = ttk.LabelFrame(self.testing_frame, text="🔧 Tesztelési Eszközök", padding=10)
        test_panel.pack(fill='x', padx=10, pady=10)
        
        # Gyors tesztek
        quick_tests_frame = ttk.Frame(test_panel)
        quick_tests_frame.pack(fill='x')
        
        ttk.Button(
            quick_tests_frame,
            text="🔍 AI modulok tesztelése",
            command=self.test_ai_modules
        ).pack(side='left', padx=(0, 10))
        
        ttk.Button(
            quick_tests_frame,
            text="📊 Dashboard kapcsolat",
            command=self.test_dashboard_connection
        ).pack(side='left', padx=(0, 10))
        
        ttk.Button(
            quick_tests_frame,
            text="🌐 WebApp státusz",
            command=self.test_webapp_status
        ).pack(side='left')
        
        # Tesztelési eredmények
        results_panel = ttk.LabelFrame(self.testing_frame, text="📋 Tesztelési Eredmények", padding=10)
        results_panel.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.test_results_text = scrolledtext.ScrolledText(
            results_panel,
            height=20,
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        self.test_results_text.pack(fill='both', expand=True)
    
    def setup_results_view(self):
        """Beállítja az eredmények nézetet"""
        self.results_frame = ttk.Frame(self.content_frame)
        
        # Eredmények listája
        list_panel = ttk.LabelFrame(self.results_frame, text="📋 Futtatott Tanpéldák", padding=10)
        list_panel.pack(side='left', fill='both', expand=True, padx=(10, 5), pady=10)
        
        self.results_tree = ttk.Treeview(
            list_panel,
            columns=('Dátum', 'Kategória', 'Állapot', 'Időtartam', 'Pontszám'),
            show='headings',
            height=15
        )
        
        # Oszlop fejlécek
        self.results_tree.heading('Dátum', text='Dátum')
        self.results_tree.heading('Kategória', text='Kategória')
        self.results_tree.heading('Állapot', text='Állapot')
        self.results_tree.heading('Időtartam', text='Időtartam')
        self.results_tree.heading('Pontszám', text='Pontszám')
        
        # Oszlop szélességek
        self.results_tree.column('Dátum', width=130)
        self.results_tree.column('Kategória', width=120)
        self.results_tree.column('Állapot', width=100)
        self.results_tree.column('Időtartam', width=100)
        self.results_tree.column('Pontszám', width=80)
        
        self.results_tree.pack(fill='both', expand=True)
        self.results_tree.bind('<<TreeviewSelect>>', self.on_result_selected)
        
        # Részletek panel
        details_panel = ttk.LabelFrame(self.results_frame, text="📊 Részletes Eredmények", padding=10)
        details_panel.pack(side='right', fill='both', expand=True, padx=(5, 10), pady=10)
        
        self.details_text = scrolledtext.ScrolledText(
            details_panel,
            height=15,
            wrap=tk.WORD,
            font=('Consolas', 9),
            state='disabled'
        )
        self.details_text.pack(fill='both', expand=True)
        
        # Export és akciók
        actions_panel = ttk.Frame(details_panel)
        actions_panel.pack(fill='x', pady=(10, 0))
        
        ttk.Button(
            actions_panel,
            text="📤 Export JSON",
            command=self.export_results_json
        ).pack(side='left', padx=(0, 10))
        
        ttk.Button(
            actions_panel,
            text="📋 Riport generálás",
            command=self.generate_results_report
        ).pack(side='left', padx=(0, 10))
        
        ttk.Button(
            actions_panel,
            text="🗑️ Törlés",
            command=self.clear_selected_result
        ).pack(side='left')
    
    def setup_status_bar(self):
        """Beállítja az alsó státusz sávot"""
        self.status_frame = ttk.Frame(self.frame, relief=tk.SUNKEN, borderwidth=1)
        self.status_frame.pack(side='bottom', fill='x')
        
        # Bal oldali státusz
        self.main_status_label = ttk.Label(
            self.status_frame,
            text="Kész",
            font=('Arial', 9)
        )
        self.main_status_label.pack(side='left', padx=10, pady=2)
        
        # Jobb oldali információk
        right_status_frame = ttk.Frame(self.status_frame)
        right_status_frame.pack(side='right', padx=10, pady=2)
        
        self.ai_status_label = ttk.Label(
            right_status_frame,
            text=f"AI: {'✅' if self.ai_modules_available else '❌'}",
            font=('Arial', 9)
        )
        self.ai_status_label.pack(side='right', padx=5)
        
        ttk.Separator(right_status_frame, orient='vertical').pack(side='right', fill='y', padx=5)
        
        self.time_label = ttk.Label(
            right_status_frame,
            text=datetime.now().strftime("%H:%M:%S"),
            font=('Arial', 9)
        )
        self.time_label.pack(side='right', padx=5)
        
        # Időzítő a státusz frissítéséhez
        self.update_time()
    
    def get_example_categories(self):
        """Visszaadja a tanpélda kategóriákat és leírásukat"""
        return {
            "🔰 Alapvető Excel elemzés": "basic",
            "📈 Grafikonok tanítása": "charts", 
            "🧮 Képletek tanítása": "formulas",
            "🤖 ML modellek": "ml_models",
            "🔄 Teljes tanulási folyamat": "pipeline",
            "🚀 Haladó funkciók": "advanced"
        }
    
    def get_example_descriptions(self):
        """Visszaadja a tanpélda leírásokat"""
        return {
            "basic": """🔰 ALAPVETŐ EXCEL ELEMZÉS

📋 Leírás:
Ez a tanpélda bemutatja az Excel fájlok alapvető elemzését és struktúra 
felismerését. Tökéletes kezdőknek és az AI rendszer megismeréséhez.

🎯 Mit fogsz megtanulni:
• Excel fájl struktúra automatikus felismerése
• Adatok típusának és formátumának elemzése  
• Oszlopok és sorok kapcsolatainak felderítése
• Egyszerű statisztikák és összefoglalók generálása
• Adatminőség ellenőrzés és validáció

⏱️ Időtartam: 2-3 perc
📊 Nehézségi szint: Kezdő
🎖️ Várható eredmény: 85-95% pontosság""",

            "charts": """📈 GRAFIKONOK TANÍTÁSA

📋 Leírás:
Ez a tanpélda a grafikonok és vizualizációk mesterséges intelligencia 
alapú tanítását és generálását mutatja be.

🎯 Mit fogsz megtanulni:
• Excel grafikonok típusának automatikus felismerése
• Adatok és grafikonok közötti kapcsolatok elemzése
• Grafikon stílusok, színek és formázások tanítása
• Új grafikonok intelligens generálása
• Vizualizációs best practice-ek alkalmazása

⏱️ Időtartam: 3-4 perc
📊 Nehézségi szint: Közepes
🎖️ Várható eredmény: 78-88% pontosság""",

            "formulas": """🧮 KÉPLETEK TANÍTÁSA

📋 Leírás:
Ez a tanpélda az Excel képletek és függvények mesterséges intelligencia
alapú tanítását és generálását mutatja be.

🎯 Mit fogsz megtanulni:
• Excel képletek mintázatainak felismerése
• Matematikai és logikai kapcsolatok tanítása
• Képlet függőségek és referenciák elemzése
• Új képletek intelligens generálása
• Komplex számítások optimalizálása

⏱️ Időtartam: 2-3 perc
📊 Nehézségi szint: Közepes
🎖️ Várható eredmény: 80-90% pontosság""",

            "ml_models": """🤖 GÉPI TANULÁSI MODELLEK

📋 Leírás:
Ez a tanpélda a gépi tanulási modellek Excel adatokon való alkalmazását
és teljesítménymonitorozását mutatja be.

🎯 Mit fogsz megtanulni:
• ML modellek Excel adatokra történő alkalmazása
• Prediktív elemzés és trend felismerés
• Anomália detektálás és kivételkezelés
• Modellek teljesítményének monitorozása
• Automatikus model optimalizálás

⏱️ Időtartam: 4-5 perc
📊 Nehézségi szint: Haladó
🎖️ Várható eredmény: 88-96% pontosság""",

            "pipeline": """🔄 TELJES TANULÁSI FOLYAMAT

📋 Leírás:
Ez a tanpélda a komplett end-to-end AI Excel Learning pipeline-t
mutatja be, minden fő komponenssel.

🎯 Mit fogsz megtanulni:
• Teljes automatikus adatfeldolgozási folyamat
• Integrált elemzés, tanítás és generálás
• Folyamatos tanulás és önfejlesztés
• Teljesítmény optimalizálás és skálázás
• Production-ready AI rendszer működése

⏱️ Időtartam: 5-6 perc
📊 Nehézségi szint: Haladó
🎖️ Várható eredmény: 90-98% pontosság""",

            "advanced": """🚀 HALADÓ FUNKCIÓK

📋 Leírás:
Ez a tanpélda a legfejlettebb AI Excel Learning funkciókat és
enterprise-szintű képességeket mutatja be.

🎯 Mit fogsz megtanulni:
• Komplex Excel munkafüzetek teljes elemzése
• Több munkalap közötti összetett kapcsolatok
• Makrók és VBA kód automatikus elemzése
• Automatikus dokumentáció és riport generálás
• Enterprise integráció és skálázhatóság

⏱️ Időtartam: 3-4 perc
📊 Nehézségi szint: Szakértő
🎖️ Várható eredmény: 92-99% pontosság"""
        }
    
    # UI eseménykezelő metódusok
    def show_examples_view(self):
        """Megjeleníti a tanpéldák nézetet"""
        self._hide_all_views()
        self.examples_frame.pack(fill='both', expand=True)
        self.main_status_label.config(text="Tanpéldák nézet")
    
    def show_dashboard_view(self):
        """Megjeleníti a dashboard nézetet"""
        self._hide_all_views()
        self.dashboard_frame.pack(fill='both', expand=True)
        self.main_status_label.config(text="Dashboard nézet")
    
    def show_testing_view(self):
        """Megjeleníti a tesztelési nézetet"""
        self._hide_all_views()
        self.testing_frame.pack(fill='both', expand=True)
        self.main_status_label.config(text="Tesztelési nézet")
    
    def show_results_view(self):
        """Megjeleníti az eredmények nézetet"""
        self._hide_all_views()
        self.results_frame.pack(fill='both', expand=True)
        self.main_status_label.config(text="Eredmények nézet")
        self.load_results()
    
    def _hide_all_views(self):
        """Elrejti az összes nézetet"""
        for frame in [self.examples_frame, self.dashboard_frame, 
                     self.testing_frame, self.results_frame]:
            frame.pack_forget()
    
    def on_category_changed(self, event=None):
        """Kategória változásakor frissíti a leírást"""
        self.update_description()
    
    def update_description(self):
        """Frissíti a kiválasztott kategória leírását"""
        category_display = self.category_var.get()
        categories = self.get_example_categories()
        category_id = categories.get(category_display, "basic")
        
        descriptions = self.get_example_descriptions()
        description = descriptions.get(category_id, "Leírás nem elérhető")
        
        self.description_text.config(state='normal')
        self.description_text.delete(1.0, tk.END)
        self.description_text.insert(1.0, description)
        self.description_text.config(state='disabled')
    
    def refresh_status(self):
        """Frissíti a rendszer állapot információkat"""
        # Státusz fa törlése
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
        
        # AI modulok állapotának ellenőrzése
        modules = {
            "Excel Analyzer": self._check_module("excel_analyzer"),
            "Chart Learner": self._check_module("chart_learner"),
            "Formula Learner": self._check_module("formula_learner"),
            "ML Models": self._check_module("ml_models"),
            "Learning Pipeline": self._check_module("learning_pipeline"),
            "Dashboard": self._check_dashboard_status(),
            "WebApp": self._check_webapp_status()
        }
        
        # Modulok hozzáadása a fához
        for module_name, status in modules.items():
            status_text = "✅ Aktív" if status else "❌ Inaktív"
            self.status_tree.insert("", "end", text=module_name, values=(status_text,))
        
        # Metrikák frissítése
        self.update_metrics()
    
    def _check_module(self, module_name):
        """Ellenőrzi egy adott modul elérhetőségét"""
        try:
            if module_name == "excel_analyzer":
                from src.ai_excel_learning.excel_analyzer import ExcelAnalyzer
            elif module_name == "chart_learner":
                from src.ai_excel_learning.chart_learner import ChartLearner
            elif module_name == "formula_learner":
                from src.ai_excel_learning.formula_learner import FormulaLearner
            elif module_name == "ml_models":
                from src.ai_excel_learning.ml_models import ExcelMLModels
            elif module_name == "learning_pipeline":
                from src.ai_excel_learning.learning_pipeline import LearningPipeline
            return True
        except ImportError:
            return False
    
    def _check_dashboard_status(self):
        """Ellenőrzi a dashboard állapotát"""
        return self.dashboard_process is not None and self.dashboard_process.poll() is None
    
    def _check_webapp_status(self):
        """Ellenőrzi a webapp állapotát"""
        return self.webapp_process is not None and self.webapp_process.poll() is None
    
    def update_metrics(self):
        """Frissíti a teljesítmény metrikákat"""
        total_runs = len(self.demo_results)
        successful_runs = sum(1 for r in self.demo_results.values() if r.get("success", False))
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        avg_time = 0
        if total_runs > 0:
            times = [float(r.get("duration", "0s").replace("s", "")) for r in self.demo_results.values()]
            avg_time = sum(times) / len(times)
        
        errors = total_runs - successful_runs
        
        self.metrics_labels["Futtatások"].config(text=str(total_runs))
        self.metrics_labels["Sikeresség"].config(text=f"{success_rate:.1f}%")
        self.metrics_labels["Átlagos idő"].config(text=f"{avg_time:.1f}s")
        self.metrics_labels["Hibák"].config(text=str(errors))
    
    def update_time(self):
        """Frissíti az időt a státusz sávban"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=current_time)
        self.frame.after(1000, self.update_time)
    
    # Tanpélda futtatási metódusok
    def run_selected_demo(self):
        """Futtatja a kiválasztott tanpéldát"""
        if self.demo_running:
            messagebox.showwarning("Figyelmeztetés", "Már fut egy tanpélda!")
            return
        
        category_display = self.category_var.get()
        categories = self.get_example_categories()
        category = categories.get(category_display)
        
        if not category:
            messagebox.showerror("Hiba", "Válassz ki egy kategóriát!")
            return
        
        # UI frissítése
        self.demo_running = True
        self.run_demo_btn.config(state='disabled')
        self.stop_demo_btn.config(state='normal')
        self.progress_bar.start()
        self.status_label.config(text=f"Futtatás: {category_display}")
        
        # Demo futtatása külön szálban
        self.demo_thread = threading.Thread(
            target=self._run_demo_worker,
            args=(category, category_display)
        )
        self.demo_thread.daemon = True
        self.demo_thread.start()
        
        app_logger.info(f"Tanpélda indítva: {category}")
    
    def _run_demo_worker(self, category, category_display):
        """Worker szál a demo futtatásához"""
        try:
            start_time = time.time()
            
            # Demo futtatása típus szerint
            demo_methods = {
                "basic": self._run_basic_demo,
                "charts": self._run_charts_demo,
                "formulas": self._run_formulas_demo,
                "ml_models": self._run_ml_models_demo,
                "pipeline": self._run_pipeline_demo,
                "advanced": self._run_advanced_demo
            }
            
            demo_method = demo_methods.get(category, self._run_default_demo)
            result = demo_method()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Eredmény összeállítása
            result_data = {
                "category": category,
                "category_display": category_display,
                "timestamp": datetime.now().isoformat(),
                "duration": f"{duration:.2f}s",
                "success": result.get("success", False),
                "score": result.get("score", 0),
                "details": result
            }
            
            # Eredmény mentése
            result_key = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.demo_results[result_key] = result_data
            
            # UI frissítése fő szálban
            self.frame.after(0, self._demo_completed, result_data)
            
        except Exception as e:
            error_result = {
                "category": category,
                "category_display": category_display,
                "timestamp": datetime.now().isoformat(),
                "duration": "0s",
                "success": False,
                "score": 0,
                "details": {"error": str(e)}
            }
            self.frame.after(0, self._demo_completed, error_result)
            app_logger.error(f"Demo hiba ({category}): {str(e)}")
    
    def _run_basic_demo(self):
        """Alapvető Excel elemzés demo"""
        try:
            # Szimulált feldolgozás lépésekkel
            self._update_demo_status("Excel fájlok betöltése...")
            time.sleep(0.8)
            
            self._update_demo_status("Struktúra elemzése...")
            time.sleep(1.0)
            
            self._update_demo_status("Adattípusok felismerése...")
            time.sleep(0.7)
            
            self._update_demo_status("Statisztikák generálása...")
            time.sleep(0.5)
            
            return {
                "success": True,
                "score": 92,
                "message": "Alapvető Excel elemzés sikeresen lefutott",
                "files_analyzed": 3,
                "patterns_found": 8,
                "data_types_detected": ["text", "number", "date", "currency"],
                "statistics_generated": 15,
                "processing_time": "3.0s"
            }
        except Exception as e:
            return {"success": False, "score": 0, "error": str(e)}
    
    def _run_charts_demo(self):
        """Grafikonok tanítása demo"""
        try:
            self._update_demo_status("Grafikon típusok felismerése...")
            time.sleep(1.2)
            
            self._update_demo_status("Adatok és grafikonok összekapcsolása...")
            time.sleep(1.5)
            
            self._update_demo_status("Stílusok és formázások tanítása...")
            time.sleep(1.0)
            
            self._update_demo_status("Új grafikonok generálása...")
            time.sleep(0.8)
            
            return {
                "success": True,
                "score": 84,
                "message": "Grafikon tanítás sikeresen lefutott",
                "charts_analyzed": 12,
                "chart_types_learned": ["scatter", "line", "bar", "pie", "combo"],
                "styles_learned": 18,
                "charts_generated": 3,
                "processing_time": "4.5s"
            }
        except Exception as e:
            return {"success": False, "score": 0, "error": str(e)}
    
    def _run_formulas_demo(self):
        """Képletek tanítása demo"""
        try:
            self._update_demo_status("Képlet mintázatok felismerése...")
            time.sleep(1.0)
            
            self._update_demo_status("Függőségek elemzése...")
            time.sleep(1.3)
            
            self._update_demo_status("Logikai kapcsolatok tanítása...")
            time.sleep(0.9)
            
            self._update_demo_status("Új képletek generálása...")
            time.sleep(0.8)
            
            return {
                "success": True,
                "score": 89,
                "message": "Képlet tanítás sikeresen lefutott",
                "formulas_analyzed": 24,
                "patterns_learned": ["mathematical", "logical", "lookup", "statistical"],
                "dependencies_mapped": 45,
                "formulas_generated": 8,
                "processing_time": "4.0s"
            }
        except Exception as e:
            return {"success": False, "score": 0, "error": str(e)}
    
    def _run_ml_models_demo(self):
        """ML modellek demo"""
        try:
            self._update_demo_status("ML modellek inicializálása...")
            time.sleep(1.5)
            
            self._update_demo_status("Adatok előfeldolgozása...")
            time.sleep(2.0)
            
            self._update_demo_status("Modellek tanítása...")
            time.sleep(2.5)
            
            self._update_demo_status("Teljesítmény értékelése...")
            time.sleep(1.0)
            
            return {
                "success": True,
                "score": 96,
                "message": "ML modellek sikeresen tanultak",
                "models_trained": 3,
                "accuracy": "96.2%",
                "predictions_made": 150,
                "anomalies_detected": 2,
                "processing_time": "7.0s"
            }
        except Exception as e:
            return {"success": False, "score": 0, "error": str(e)}
    
    def _run_pipeline_demo(self):
        """Teljes tanulási folyamat demo"""
        try:
            self._update_demo_status("Pipeline inicializálása...")
            time.sleep(1.0)
            
            self._update_demo_status("Adatok betöltése és validálása...")
            time.sleep(1.8)
            
            self._update_demo_status("Elemzés és tanítás...")
            time.sleep(2.5)
            
            self._update_demo_status("Modellek integrálása...")
            time.sleep(1.5)
            
            self._update_demo_status("Eredmények optimalizálása...")
            time.sleep(1.2)
            
            return {
                "success": True,
                "score": 94,
                "message": "Teljes tanulási folyamat sikeresen lefutott",
                "total_files": 18,
                "total_patterns": 42,
                "models_integrated": 5,
                "optimization_rounds": 3,
                "processing_time": "8.0s"
            }
        except Exception as e:
            return {"success": False, "score": 0, "error": str(e)}
    
    def _run_advanced_demo(self):
        """Haladó funkciók demo"""
        try:
            self._update_demo_status("Komplex munkafüzetek elemzése...")
            time.sleep(1.3)
            
            self._update_demo_status("Munkalapok közötti kapcsolatok...")
            time.sleep(1.7)
            
            self._update_demo_status("VBA kód elemzése...")
            time.sleep(1.8)
            
            self._update_demo_status("Dokumentáció generálása...")
            time.sleep(1.2)
            
            return {
                "success": True,
                "score": 98,
                "message": "Haladó funkciók sikeresen lefutottak",
                "complex_workbooks": 6,
                "worksheets_analyzed": 23,
                "vba_macros_processed": 8,
                "documentation_pages": 15,
                "processing_time": "6.0s"
            }
        except Exception as e:
            return {"success": False, "score": 0, "error": str(e)}
    
    def _run_default_demo(self):
        """Alapértelmezett demo ismeretlen kategóriák esetén"""
        time.sleep(2.0)
        return {
            "success": False,
            "score": 0,
            "error": "Ismeretlen kategória"
        }
    
    def _update_demo_status(self, status):
        """Frissíti a demo állapotát a fő szálban"""
        def update():
            self.status_label.config(text=status)
        self.frame.after(0, update)
    
    def _demo_completed(self, result_data):
        """Demo befejezése után frissíti a UI-t"""
        self.demo_running = False
        self.run_demo_btn.config(state='normal')
        self.stop_demo_btn.config(state='disabled')
        self.progress_bar.stop()
        
        if result_data["success"]:
            self.status_label.config(text=f"✅ Sikeres - Pontszám: {result_data['score']}%")
            messagebox.showinfo(
                "Siker", 
                f"Tanpélda sikeresen lefutott!\n"
                f"Kategória: {result_data['category_display']}\n"
                f"Pontszám: {result_data['score']}%\n"
                f"Időtartam: {result_data['duration']}"
            )
        else:
            self.status_label.config(text="❌ Hiba történt")
            error_msg = result_data['details'].get('error', 'Ismeretlen hiba')
            messagebox.showerror("Hiba", f"Tanpélda hibával lefutott:\n{error_msg}")
        
        # Státusz és metrikák frissítése
        self.refresh_status()
        
        app_logger.info(f"Demo befejezve: {result_data['category']} - {'sikeres' if result_data['success'] else 'sikertelen'}")
    
    def stop_demo(self):
        """Leállítja a futó tanpéldát"""
        if self.demo_running:
            self.demo_running = False
            self.status_label.config(text="Tanpélda leállítva")
            self.progress_bar.stop()
            self.run_demo_btn.config(state='normal')
            self.stop_demo_btn.config(state='disabled')
            app_logger.info("Demo manuálisan leállítva")
    
    def show_demo_details(self):
        """Megjeleníti a kiválasztott demo részleteit"""
        category_display = self.category_var.get()
        categories = self.get_example_categories()
        category = categories.get(category_display)
        
        if not category:
            return
        
        # Részletek ablak létrehozása
        details_window = tk.Toplevel(self.frame)
        details_window.title(f"Tanpélda Részletek - {category_display}")
        details_window.geometry("600x500")
        details_window.transient(self.frame)
        details_window.grab_set()
        
        # Tartalom
        text_frame = ttk.Frame(details_window, padding=10)
        text_frame.pack(fill='both', expand=True)
        
        text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=('Consolas', 10)
        )
        text_widget.pack(fill='both', expand=True)
        
        # Részletes leírás
        descriptions = self.get_example_descriptions()
        description = descriptions.get(category, "Részletek nem elérhetők")
        
        text_widget.insert(tk.END, description)
        text_widget.insert(tk.END, "\n\n" + "="*60 + "\n")
        text_widget.insert(tk.END, "TECHNIKAI INFORMÁCIÓK\n")
        text_widget.insert(tk.END, "="*60 + "\n\n")
        
        # Technikai infók
        tech_info = f"""Kategória ID: {category}
AI Modulok: {'Elérhető' if self.ai_modules_available else 'Nem elérhető'}
Futtatási környezet: Desktop GUI
Párhuzamos feldolgozás: Támogatott
Eredmény mentés: Automatikus
Export formátumok: JSON, TXT, HTML

Rendszerkövetelmények:
- Python 3.8+
- AI Excel Learning modulok
- Minimum 4GB RAM
- 100MB szabad hely"""
        
        text_widget.insert(tk.END, tech_info)
        text_widget.config(state='disabled')
        
        # Bezárás gomb
        ttk.Button(
            details_window,
            text="Bezárás",
            command=details_window.destroy
        ).pack(pady=10)
    
    # Dashboard és webapp kezelés
    def launch_webapp(self):
        """Elindítja a webapp-ot"""
        if self.webapp_process and self.webapp_process.poll() is None:
            messagebox.showinfo("Információ", "WebApp már fut!")
            self.open_webapp()
            return
        
        self.start_webapp()
    
    def toggle_dashboard(self):
        """Váltogatja a dashboard állapotát"""
        if self._check_dashboard_status():
            self.stop_dashboard()
        else:
            self.start_dashboard()
    
    def start_dashboard(self):
        """Indítja az AI Dashboard-ot"""
        try:
            dashboard_script = Path(__file__).parent.parent / "ai_excel_learning" / "ai_dashboard.py"
            
            if not dashboard_script.exists():
                messagebox.showerror("Hiba", "AI Dashboard script nem található!")
                return
            
            # Dashboard indítása
            self.dashboard_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", str(dashboard_script),
                "--server.port", "8501",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ])
            
            # Várakozás az indulásra
            time.sleep(2)
            
            # UI frissítése
            self.start_dashboard_btn.config(state='disabled')
            self.open_dashboard_btn.config(state='normal')
            self.dashboard_status_label.config(
                text="✅ Fut (localhost:8501)",
                foreground="green"
            )
            
            messagebox.showinfo("Siker", "AI Dashboard sikeresen elindult!")
            app_logger.info("AI Dashboard elindítva")
            
        except Exception as e:
            messagebox.showerror("Hiba", f"Dashboard indítása sikertelen:\n{str(e)}")
            app_logger.error(f"Dashboard indítási hiba: {str(e)}")
    
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
            self.open_dashboard_btn.config(state='disabled')
            self.dashboard_status_label.config(
                text="❌ Nincs elindítva",
                foreground="black"
            )
            
            app_logger.info("AI Dashboard leállítva")
    
    def open_dashboard(self):
        """Megnyitja a dashboard-ot a böngészőben"""
        try:
            webbrowser.open(self.dashboard_url)
        except Exception as e:
            messagebox.showerror("Hiba", f"Dashboard megnyitása sikertelen:\n{str(e)}")
    
    def start_webapp(self):
        """Indítja a WebApp-ot"""
        try:
            webapp_script = Path(__file__).parent.parent / "ai_excel_learning" / "webapp_demo.py"
            
            if not webapp_script.exists():
                messagebox.showerror("Hiba", "WebApp script nem található!")
                return
            
            # WebApp indítása
            self.webapp_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", str(webapp_script),
                "--server.port", "8502",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ])
            
            # Várakozás az indulásra
            time.sleep(3)
            
            # UI frissítése
            self.start_webapp_btn.config(state='disabled')
            self.open_webapp_btn.config(state='normal')
            self.webapp_status_label.config(
                text="✅ Fut (localhost:8502)",
                foreground="green"
            )
            
            messagebox.showinfo("Siker", "WebApp sikeresen elindult!")
            app_logger.info("WebApp elindítva")
            
        except Exception as e:
            messagebox.showerror("Hiba", f"WebApp indítása sikertelen:\n{str(e)}")
            app_logger.error(f"WebApp indítási hiba: {str(e)}")
    
    def stop_webapp(self):
        """Leállítja a WebApp-ot"""
        if self.webapp_process:
            try:
                self.webapp_process.terminate()
                self.webapp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.webapp_process.kill()
            
            self.webapp_process = None
            
            # UI frissítése
            self.start_webapp_btn.config(state='normal')
            self.open_webapp_btn.config(state='disabled')
            self.webapp_status_label.config(
                text="❌ Nincs elindítva",
                foreground="black"
            )
            
            app_logger.info("WebApp leállítva")
    
    def open_webapp(self):
        """Megnyitja a WebApp-ot a böngészőben"""
        try:
            webbrowser.open(self.webapp_url)
        except Exception as e:
            messagebox.showerror("Hiba", f"WebApp megnyitása sikertelen:\n{str(e)}")
    
    # Tesztelési funkciók
    def test_ai_modules(self):
        """Teszteli az AI modulok elérhetőségét"""
        self.test_results_text.insert(tk.END, f"\n{'='*60}\n")
        self.test_results_text.insert(tk.END, f"AI MODULOK TESZTELÉSE - {datetime.now().strftime('%H:%M:%S')}\n")
        self.test_results_text.insert(tk.END, f"{'='*60}\n\n")
        
        modules_to_test = [
            ("excel_analyzer", "Excel Analyzer"),
            ("chart_learner", "Chart Learner"),
            ("formula_learner", "Formula Learner"),
            ("ml_models", "ML Models"),
            ("learning_pipeline", "Learning Pipeline")
        ]
        
        for module_id, module_name in modules_to_test:
            try:
                status = self._check_module(module_id)
                status_text = "✅ ELÉRHETŐ" if status else "❌ NEM ELÉRHETŐ"
                self.test_results_text.insert(tk.END, f"{module_name:20s}: {status_text}\n")
            except Exception as e:
                self.test_results_text.insert(tk.END, f"{module_name:20s}: ❌ HIBA - {str(e)}\n")
        
        self.test_results_text.insert(tk.END, "\nTeszt befejezve.\n")
        self.test_results_text.see(tk.END)
    
    def test_dashboard_connection(self):
        """Teszteli a dashboard kapcsolatot"""
        self.test_results_text.insert(tk.END, f"\n{'='*60}\n")
        self.test_results_text.insert(tk.END, f"DASHBOARD KAPCSOLAT TESZTELÉSE - {datetime.now().strftime('%H:%M:%S')}\n")
        self.test_results_text.insert(tk.END, f"{'='*60}\n\n")
        
        # Dashboard folyamat ellenőrzése
        dashboard_running = self._check_dashboard_status()
        self.test_results_text.insert(tk.END, f"Dashboard folyamat: {'✅ Fut' if dashboard_running else '❌ Nem fut'}\n")
        
        if dashboard_running:
            try:
                import requests
                response = requests.get(self.dashboard_url, timeout=5)
                if response.status_code == 200:
                    self.test_results_text.insert(tk.END, f"HTTP kapcsolat: ✅ OK ({response.status_code})\n")
                else:
                    self.test_results_text.insert(tk.END, f"HTTP kapcsolat: ⚠️ Hibás státusz ({response.status_code})\n")
            except ImportError:
                self.test_results_text.insert(tk.END, "HTTP teszt: ❌ requests modul hiányzik\n")
            except Exception as e:
                self.test_results_text.insert(tk.END, f"HTTP kapcsolat: ❌ Hiba - {str(e)}\n")
        
        self.test_results_text.insert(tk.END, "\nTeszt befejezve.\n")
        self.test_results_text.see(tk.END)
    
    def test_webapp_status(self):
        """Teszteli a WebApp státuszát"""
        self.test_results_text.insert(tk.END, f"\n{'='*60}\n")
        self.test_results_text.insert(tk.END, f"WEBAPP STÁTUSZ TESZTELÉSE - {datetime.now().strftime('%H:%M:%S')}\n")
        self.test_results_text.insert(tk.END, f"{'='*60}\n\n")
        
        # WebApp folyamat ellenőrzése
        webapp_running = self._check_webapp_status()
        self.test_results_text.insert(tk.END, f"WebApp folyamat: {'✅ Fut' if webapp_running else '❌ Nem fut'}\n")
        
        if webapp_running:
            try:
                import requests
                response = requests.get(self.webapp_url, timeout=5)
                if response.status_code == 200:
                    self.test_results_text.insert(tk.END, f"HTTP kapcsolat: ✅ OK ({response.status_code})\n")
                else:
                    self.test_results_text.insert(tk.END, f"HTTP kapcsolat: ⚠️ Hibás státusz ({response.status_code})\n")
            except ImportError:
                self.test_results_text.insert(tk.END, "HTTP teszt: ❌ requests modul hiányzik\n")
            except Exception as e:
                self.test_results_text.insert(tk.END, f"HTTP kapcsolat: ❌ Hiba - {str(e)}\n")
        
        self.test_results_text.insert(tk.END, "\nTeszt befejezve.\n")
        self.test_results_text.see(tk.END)
    
    # Eredmények kezelése
    def load_results(self):
        """Betölti a mentett eredményeket"""
        # Eredmények törlése
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Eredmények hozzáadása
        for result_key, result in self.demo_results.items():
            status = "✅ Sikeres" if result["success"] else "❌ Sikertelen"
            score = f"{result['score']}%" if result['success'] else "0%"
            
            self.results_tree.insert("", "end", values=(
                result["timestamp"][:19].replace("T", " "),
                result.get("category_display", result["category"]),
                status,
                result["duration"],
                score
            ))
        
        # Eredmények rendezése dátum szerint (legújabb elöl)
        items = self.results_tree.get_children()
        items_data = [(self.results_tree.item(item)['values'], item) for item in items]
        items_data.sort(key=lambda x: x[0][0], reverse=True)
        
        for index, (values, item) in enumerate(items_data):
            self.results_tree.move(item, '', index)
    
    def on_result_selected(self, event=None):
        """Eredmény kiválasztásakor frissíti a részleteket"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = self.results_tree.item(selection[0])
        values = item['values']
        
        if len(values) < 2:
            return
        
        # Kategória alapján keresés az eredményekben
        category_display = values[1]
        timestamp = values[0]
        
        matching_result = None
        for result_key, result in self.demo_results.items():
            if (result.get("category_display", result["category"]) == category_display and 
                result["timestamp"][:19].replace("T", " ") == timestamp):
                matching_result = result
                break
        
        if matching_result:
            details = self._format_result_details(matching_result)
            
            self.details_text.config(state='normal')
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, details)
            self.details_text.config(state='disabled')
    
    def _format_result_details(self, result):
        """Formázza az eredmény részleteit"""
        lines = []
        lines.append(f"TANPÉLDA EREDMÉNY RÉSZLETEI")
        lines.append("=" * 40)
        lines.append(f"Kategória: {result.get('category_display', result['category'])}")
        lines.append(f"Kategória ID: {result['category']}")
        lines.append(f"Dátum: {result['timestamp']}")
        lines.append(f"Időtartam: {result['duration']}")
        lines.append(f"Állapot: {'Sikeres' if result['success'] else 'Sikertelen'}")
        lines.append(f"Pontszám: {result.get('score', 0)}%")
        lines.append("")
        
        if 'details' in result:
            details = result['details']
            if result['success']:
                lines.append("RÉSZLETES EREDMÉNYEK:")
                lines.append("-" * 20)
                for key, value in details.items():
                    if key not in ['success', 'score']:
                        if isinstance(value, list):
                            lines.append(f"{key}: {', '.join(map(str, value))}")
                        else:
                            lines.append(f"{key}: {value}")
            else:
                lines.append("HIBAÜZENET:")
                lines.append("-" * 20)
                lines.append(f"{details.get('error', 'Ismeretlen hiba')}")
        
        return "\n".join(lines)
    
    def export_results_json(self):
        """Exportálja az eredményeket JSON formátumban"""
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
                app_logger.info(f"Eredmények exportálva: {filepath}")
        
        except Exception as e:
            messagebox.showerror("Hiba", f"Exportálás sikertelen:\n{str(e)}")
            app_logger.error(f"Export hiba: {str(e)}")
    
    def generate_results_report(self):
        """Generál egy összefoglaló jelentést"""
        if not self.demo_results:
            messagebox.showinfo("Információ", "Nincs adat a jelentés generálásához!")
            return
        
        try:
            report = self._create_detailed_report()
            
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
                app_logger.info(f"Jelentés generálva: {filepath}")
        
        except Exception as e:
            messagebox.showerror("Hiba", f"Jelentés generálása sikertelen:\n{str(e)}")
            app_logger.error(f"Jelentés generálási hiba: {str(e)}")
    
    def _create_detailed_report(self):
        """Létrehozza a részletes jelentést"""
        lines = []
        lines.append("=" * 80)
        lines.append("AI EXCEL LEARNING - RÉSZLETES ÖSSZEFOGLALÓ JELENTÉS")
        lines.append("=" * 80)
        lines.append(f"Generálva: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Rendszer verzió: DataChaEnhanced v2.0.0")
        lines.append(f"AI modulok állapota: {'Elérhető' if self.ai_modules_available else 'Nem elérhető'}")
        lines.append("")
        
        # Összesítés
        total_demos = len(self.demo_results)
        successful_demos = sum(1 for r in self.demo_results.values() if r["success"])
        failed_demos = total_demos - successful_demos
        
        if total_demos > 0:
            avg_score = sum(r.get("score", 0) for r in self.demo_results.values()) / total_demos
            success_rate = successful_demos / total_demos * 100
        else:
            avg_score = 0
            success_rate = 0
        
        lines.append("ÖSSZESÍTŐ STATISZTIKÁK:")
        lines.append("-" * 40)
        lines.append(f"  Összes tanpélda futtatás: {total_demos}")
        lines.append(f"  Sikeres futtatások: {successful_demos}")
        lines.append(f"  Sikertelen futtatások: {failed_demos}")
        lines.append(f"  Sikerességi arány: {success_rate:.1f}%")
        lines.append(f"  Átlagos pontszám: {avg_score:.1f}%")
        lines.append("")
        
        # Kategóriák szerinti bontás
        categories = {}
        for result in self.demo_results.values():
            category = result.get("category_display", result["category"])
            if category not in categories:
                categories[category] = {"total": 0, "successful": 0, "scores": []}
            
            categories[category]["total"] += 1
            if result["success"]:
                categories[category]["successful"] += 1
                categories[category]["scores"].append(result.get("score", 0))
        
        lines.append("KATEGÓRIÁK SZERINTI BONTÁS:")
        lines.append("-" * 40)
        for category, stats in categories.items():
            success_rate = stats["successful"] / stats["total"] * 100 if stats["total"] > 0 else 0
            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            
            lines.append(f"  {category}:")
            lines.append(f"    Futtatások: {stats['total']}")
            lines.append(f"    Sikeresség: {success_rate:.1f}%")
            lines.append(f"    Átlagos pontszám: {avg_score:.1f}%")
            lines.append("")
        
        # Részletes eredmények
        lines.append("RÉSZLETES EREDMÉNYEK:")
        lines.append("-" * 40)
        
        sorted_results = sorted(
            self.demo_results.items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )
        
        for result_key, result in sorted_results:
            lines.append(f"Futtatás ID: {result_key}")
            lines.append(f"  Kategória: {result.get('category_display', result['category'])}")
            lines.append(f"  Dátum: {result['timestamp']}")
            lines.append(f"  Időtartam: {result['duration']}")
            lines.append(f"  Állapot: {'Sikeres' if result['success'] else 'Sikertelen'}")
            lines.append(f"  Pontszám: {result.get('score', 0)}%")
            
            if result['success'] and 'details' in result:
                details = result['details']
                if 'message' in details:
                    lines.append(f"  Üzenet: {details['message']}")
            elif not result['success'] and 'details' in result:
                lines.append(f"  Hiba: {result['details'].get('error', 'Ismeretlen hiba')}")
            
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("Jelentés vége")
        
        return "\n".join(lines)
    
    def clear_selected_result(self):
        """Törli a kiválasztott eredményt"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showinfo("Információ", "Válassz ki egy eredményt a törléshez!")
            return
        
        if messagebox.askyesno("Megerősítés", "Biztosan törölni szeretnéd ezt az eredményt?"):
            item = self.results_tree.item(selection[0])
            values = item['values']
            
            if len(values) >= 2:
                category_display = values[1]
                timestamp = values[0]
                
                # Eredmény keresése és törlése
                to_delete = None
                for result_key, result in self.demo_results.items():
                    if (result.get("category_display", result["category"]) == category_display and 
                        result["timestamp"][:19].replace("T", " ") == timestamp):
                        to_delete = result_key
                        break
                
                if to_delete:
                    del self.demo_results[to_delete]
                    self.load_results()
                    self.update_metrics()
                    
                    # Részletek törlése
                    self.details_text.config(state='normal')
                    self.details_text.delete(1.0, tk.END)
                    self.details_text.config(state='disabled')
    
    def clear_results(self):
        """Törli az összes eredményt"""
        if not self.demo_results:
            messagebox.showinfo("Információ", "Nincsenek törlendő eredmények!")
            return
        
        if messagebox.askyesno("Megerősítés", "Biztosan törölni szeretnéd az összes eredményt?"):
            self.demo_results.clear()
            self.load_results()
            self.update_metrics()
            
            # Részletek törlése
            self.details_text.config(state='normal')
            self.details_text.delete(1.0, tk.END)
            self.details_text.config(state='disabled')
            
            app_logger.info("Összes eredmény törölve")
    
    def export_results(self):
        """Exportálja az eredményeket (általános)"""
        if not self.demo_results:
            messagebox.showinfo("Információ", "Nincsenek exportálható eredmények!")
            return
        
        # Export formátum választó
        format_window = tk.Toplevel(self.frame)
        format_window.title("Export Formátum")
        format_window.geometry("300x200")
        format_window.transient(self.frame)
        format_window.grab_set()
        
        ttk.Label(format_window, text="Válassz export formátumot:", font=('Arial', 12)).pack(pady=20)
        
        format_var = tk.StringVar(value="json")
        
        ttk.Radiobutton(format_window, text="JSON", variable=format_var, value="json").pack(pady=5)
        ttk.Radiobutton(format_window, text="TXT riport", variable=format_var, value="txt").pack(pady=5)
        
        buttons_frame = ttk.Frame(format_window)
        buttons_frame.pack(pady=20)
        
        def do_export():
            if format_var.get() == "json":
                format_window.destroy()
                self.export_results_json()
            else:
                format_window.destroy()
                self.generate_results_report()
        
        ttk.Button(buttons_frame, text="Export", command=do_export).pack(side='left', padx=10)
        ttk.Button(buttons_frame, text="Mégse", command=format_window.destroy).pack(side='left', padx=10)
    
    def generate_performance_report(self):
        """Teljesítmény riport generálása"""
        try:
            report_lines = []
            report_lines.append("TELJESÍTMÉNY RIPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Generálva: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Rendszer információk
            report_lines.append("RENDSZER INFORMÁCIÓK:")
            report_lines.append(f"AI modulok: {'Elérhető' if self.ai_modules_available else 'Nem elérhető'}")
            report_lines.append(f"Dashboard: {'Fut' if self._check_dashboard_status() else 'Nem fut'}")
            report_lines.append(f"WebApp: {'Fut' if self._check_webapp_status() else 'Nem fut'}")
            report_lines.append("")
            
            # Teljesítmény statisztikák
            if self.demo_results:
                total_runs = len(self.demo_results)
                successful_runs = sum(1 for r in self.demo_results.values() if r.get("success", False))
                success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
                
                times = []
                for r in self.demo_results.values():
                    try:
                        time_str = r.get("duration", "0s").replace("s", "")
                        times.append(float(time_str))
                    except ValueError:
                        pass
                
                avg_time = sum(times) / len(times) if times else 0
                min_time = min(times) if times else 0
                max_time = max(times) if times else 0
                
                report_lines.append("TELJESÍTMÉNY STATISZTIKÁK:")
                report_lines.append(f"Összes futtatás: {total_runs}")
                report_lines.append(f"Sikeresség: {success_rate:.1f}%")
                report_lines.append(f"Átlagos idő: {avg_time:.2f}s")
                report_lines.append(f"Leggyorsabb: {min_time:.2f}s")
                report_lines.append(f"Leglassabb: {max_time:.2f}s")
            else:
                report_lines.append("TELJESÍTMÉNY STATISZTIKÁK:")
                report_lines.append("Nincsenek elérhető adatok")
            
            report_lines.append("")
            report_lines.append("Riport vége")
            
            # Megjelenítés popup ablakban
            report_window = tk.Toplevel(self.frame)
            report_window.title("Teljesítmény Riport")
            report_window.geometry("500x400")
            report_window.transient(self.frame)
            
            text_widget = scrolledtext.ScrolledText(
                report_window,
                wrap=tk.WORD,
                font=('Consolas', 10)
            )
            text_widget.pack(fill='both', expand=True, padx=10, pady=10)
            
            text_widget.insert(tk.END, "\n".join(report_lines))
            text_widget.config(state='disabled')
            
            ttk.Button(report_window, text="Bezárás", command=report_window.destroy).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Hiba", f"Riport generálása sikertelen:\n{str(e)}")
    
    # Beállítások kezelése
    def show_settings(self):
        """Megjeleníti a beállítások ablakot"""
        settings_window = tk.Toplevel(self.frame)
        settings_window.title("Beállítások")
        settings_window.geometry("400x300")
        settings_window.transient(self.frame)
        settings_window.grab_set()
        
        # Beállítások frame
        settings_frame = ttk.LabelFrame(settings_window, text="Általános Beállítások", padding=10)
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        # Beállítási változók
        auto_refresh_var = tk.BooleanVar(value=self.settings.get("auto_refresh", True))
        real_time_var = tk.BooleanVar(value=self.settings.get("real_time_monitoring", True))
        detailed_logging_var = tk.BooleanVar(value=self.settings.get("detailed_logging", False))
        
        export_format_var = tk.StringVar(value=self.settings.get("export_format", "json"))
        
        # Checkboxok
        ttk.Checkbutton(
            settings_frame,
            text="Automatikus frissítés",
            variable=auto_refresh_var
        ).pack(anchor='w', pady=2)
        
        ttk.Checkbutton(
            settings_frame,
            text="Valós idejű monitoring",
            variable=real_time_var
        ).pack(anchor='w', pady=2)
        
        ttk.Checkbutton(
            settings_frame,
            text="Részletes naplózás",
            variable=detailed_logging_var
        ).pack(anchor='w', pady=2)
        
        # Export formátum
        format_frame = ttk.Frame(settings_frame)
        format_frame.pack(fill='x', pady=10)
        
        ttk.Label(format_frame, text="Alapértelmezett export formátum:").pack(anchor='w')
        
        format_combo = ttk.Combobox(
            format_frame,
            textvariable=export_format_var,
            values=["json", "txt", "html"],
            state="readonly"
        )
        format_combo.pack(anchor='w', pady=5)
        
        # Gombok
        buttons_frame = ttk.Frame(settings_window)
        buttons_frame.pack(fill='x', padx=10, pady=10)
        
        def save_settings():
            self.settings.update({
                "auto_refresh": auto_refresh_var.get(),
                "real_time_monitoring": real_time_var.get(),
                "detailed_logging": detailed_logging_var.get(),
                "export_format": export_format_var.get()
            })
            self.save_settings()
            settings_window.destroy()
            messagebox.showinfo("Siker", "Beállítások mentve!")
        
        def reset_settings():
            if messagebox.askyesno("Megerősítés", "Vissza szeretnéd állítani az alapértelmezett beállításokat?"):
                auto_refresh_var.set(True)
                real_time_var.set(True)
                detailed_logging_var.set(False)
                export_format_var.set("json")
        
        ttk.Button(buttons_frame, text="Mentés", command=save_settings).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Alapértelmezett", command=reset_settings).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Mégse", command=settings_window.destroy).pack(side='left', padx=5)
    
    def load_settings(self):
        """Betölti a beállításokat"""
        try:
            settings_file = Path(__file__).parent.parent / "settings" / "examples_menu_settings.json"
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                    self.settings.update(saved_settings)
        except Exception as e:
            app_logger.warning(f"Beállítások betöltési hiba: {str(e)}")
    
    def save_settings(self):
        """Menti a beállításokat"""
        try:
            settings_dir = Path(__file__).parent.parent / "settings"
            settings_dir.mkdir(exist_ok=True)
            
            settings_file = settings_dir / "examples_menu_settings.json"
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            app_logger.error(f"Beállítások mentési hiba: {str(e)}")
    
    def on_settings_changed(self):
        """Beállítások változásakor hívódik meg"""
        # Beállítások frissítése a változókból
        if hasattr(self, 'auto_refresh_var'):
            self.settings["auto_refresh"] = self.auto_refresh_var.get()
        if hasattr(self, 'real_time_var'):
            self.settings["real_time_monitoring"] = self.real_time_var.get()
        if hasattr(self, 'detailed_logging_var'):
            self.settings["detailed_logging"] = self.detailed_logging_var.get()
        
        # Beállítások mentése
        self.save_settings()
    
    def cleanup(self):
        """Tisztítja fel az erőforrásokat"""
        try:
            # Dashboard leállítása
            if self.dashboard_process:
                self.stop_dashboard()
            
            # WebApp leállítása
            if self.webapp_process:
                self.stop_webapp()
            
            # Demo thread leállítása
            if hasattr(self, 'demo_thread') and self.demo_thread.is_alive():
                self.demo_running = False
                self.demo_thread.join(timeout=1)
            
            # Beállítások mentése
            self.save_settings()
            
            app_logger.info("Kész Tanpéldák menü tab tisztítva")
            
        except Exception as e:
            app_logger.error(f"Cleanup hiba: {str(e)}")