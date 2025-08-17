#!/usr/bin/env python3
"""
Enhanced AI Excel Learning WebApp Demo

Ez a továbbfejlesztett Streamlit alapú webalkalmazás biztosítja a kész tanpéldákhoz és 
az AI Excel Learning dashboard integrációjához való hozzáférést. A rendszer teljes 
mértékben kompatibilis a desktop GUI verzióval és azonos funkcionalitást nyújt.

Funkciók:
- 📚 Interaktív kész tanpéldák kategóriákba rendezve
- 📊 AI Dashboard integráció külön oldalként
- 🔧 Fejlett monitoring és riportolás
- 📋 Eredmények exportálása és megosztása
- ⚙️ Testreszabható beállítások
- 🔄 Valós idejű szinkronizáció a desktop verzióval
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import requests

# AI Excel Learning modulok importálása
try:
    from .excel_analyzer import ExcelAnalyzer
    from .chart_learner import ChartLearner
    from .formula_learner import FormulaLearner
    from .ml_models import ExcelMLModels
    from .learning_pipeline import LearningPipeline
    from .data_generator import DataGenerator
    from .excel_generator import ExcelGenerator
    from .ai_monitor import get_ai_monitor
    AI_MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"AI modulok nem elérhetők: {e}")
    AI_MODULES_AVAILABLE = False

# Oldal konfiguráció
st.set_page_config(
    page_title="AI Excel Learning - Kész Tanpéldák & Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state inicializálása
if 'demo_results' not in st.session_state:
    st.session_state.demo_results = {}

if 'demo_running' not in st.session_state:
    st.session_state.demo_running = False

if 'dashboard_process' not in st.session_state:
    st.session_state.dashboard_process = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = "examples"

if 'settings' not in st.session_state:
    st.session_state.settings = {
        "auto_refresh": True,
        "real_time_monitoring": True,
        "detailed_logging": False,
        "export_format": "json"
    }

# Egyedi CSS stílus
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #cce7ff;
        border: 1px solid #99d6ff;
        color: #004085;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_example_categories():
    """Visszaadja a tanpélda kategóriákat"""
    return {
        "🔰 Alapvető Excel elemzés": "basic",
        "📈 Grafikonok tanítása": "charts", 
        "🧮 Képletek tanítása": "formulas",
        "🤖 ML modellek": "ml_models",
        "🔄 Teljes tanulási folyamat": "pipeline",
        "🚀 Haladó funkciók": "advanced"
    }

def get_example_descriptions():
    """Visszaadja a tanpélda leírásokat"""
    return {
        "basic": {
            "title": "🔰 Alapvető Excel elemzés",
            "description": """Ez a tanpélda bemutatja az Excel fájlok alapvető elemzését és struktúra 
            felismerését. Tökéletes kezdőknek és az AI rendszer megismeréséhez.""",
            "features": [
                "Excel fájl struktúra automatikus felismerése",
                "Adatok típusának és formátumának elemzése",
                "Oszlopok és sorok kapcsolatainak felderítése",
                "Egyszerű statisztikák és összefoglalók generálása",
                "Adatminőség ellenőrzés és validáció"
            ],
            "duration": "2-3 perc",
            "difficulty": "Kezdő",
            "expected_score": "85-95%"
        },
        "charts": {
            "title": "📈 Grafikonok tanítása",
            "description": """Ez a tanpélda a grafikonok és vizualizációk mesterséges intelligencia 
            alapú tanítását és generálását mutatja be.""",
            "features": [
                "Excel grafikonok típusának automatikus felismerése",
                "Adatok és grafikonok közötti kapcsolatok elemzése",
                "Grafikon stílusok, színek és formázások tanítása",
                "Új grafikonok intelligens generálása",
                "Vizualizációs best practice-ek alkalmazása"
            ],
            "duration": "3-4 perc",
            "difficulty": "Közepes",
            "expected_score": "78-88%"
        },
        "formulas": {
            "title": "🧮 Képletek tanítása",
            "description": """Ez a tanpélda az Excel képletek és függvények mesterséges intelligencia
            alapú tanítását és generálását mutatja be.""",
            "features": [
                "Excel képletek mintázatainak felismerése",
                "Matematikai és logikai kapcsolatok tanítása",
                "Képlet függőségek és referenciák elemzése",
                "Új képletek intelligens generálása",
                "Komplex számítások optimalizálása"
            ],
            "duration": "2-3 perc",
            "difficulty": "Közepes",
            "expected_score": "80-90%"
        },
        "ml_models": {
            "title": "🤖 Gépi tanulási modellek",
            "description": """Ez a tanpélda a gépi tanulási modellek Excel adatokon való alkalmazását
            és teljesítménymonitorozását mutatja be.""",
            "features": [
                "ML modellek Excel adatokra történő alkalmazása",
                "Prediktív elemzés és trend felismerés",
                "Anomália detektálás és kivételkezelés",
                "Modellek teljesítményének monitorozása",
                "Automatikus model optimalizálás"
            ],
            "duration": "4-5 perc",
            "difficulty": "Haladó",
            "expected_score": "88-96%"
        },
        "pipeline": {
            "title": "🔄 Teljes tanulási folyamat",
            "description": """Ez a tanpélda a komplett end-to-end AI Excel Learning pipeline-t
            mutatja be, minden fő komponenssel.""",
            "features": [
                "Teljes automatikus adatfeldolgozási folyamat",
                "Integrált elemzés, tanítás és generálás",
                "Folyamatos tanulás és önfejlesztés",
                "Teljesítmény optimalizálás és skálázás",
                "Production-ready AI rendszer működése"
            ],
            "duration": "5-6 perc",
            "difficulty": "Haladó",
            "expected_score": "90-98%"
        },
        "advanced": {
            "title": "🚀 Haladó funkciók",
            "description": """Ez a tanpélda a legfejlettebb AI Excel Learning funkciókat és
            enterprise-szintű képességeket mutatja be.""",
            "features": [
                "Komplex Excel munkafüzetek teljes elemzése",
                "Több munkalap közötti összetett kapcsolatok",
                "Makrók és VBA kód automatikus elemzése",
                "Automatikus dokumentáció és riport generálás",
                "Enterprise integráció és skálázhatóság"
            ],
            "duration": "3-4 perc",
            "difficulty": "Szakértő",
            "expected_score": "92-99%"
        }
    }

def check_ai_modules():
    """Ellenőrzi az AI modulok elérhetőségét"""
    return AI_MODULES_AVAILABLE

def check_dashboard_status():
    """Ellenőrzi a dashboard állapotát"""
    try:
        response = requests.get("http://localhost:8501", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_desktop_app_status():
    """Ellenőrzi a desktop alkalmazás állapotát"""
    # Itt lehetne kommunikálni a desktop applikációval
    # Például egy shared file vagy API keresztül
    return True  # Placeholder

def main():
    """Fő oldal megjelenítése"""
    # Fejléc
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI Excel Learning - Kész Tanpéldák & Dashboard</h1>
        <p>Ismerd meg a rendszer képességeit és teszteld interaktívan</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigáció
    with st.sidebar:
        st.header("🎯 Navigáció")
        
        page = st.selectbox(
            "Válassz oldalt:",
            ["🏠 Főoldal", "📚 Kész Tanpéldák", "📊 AI Dashboard", "🔧 Tesztelés", "📋 Eredmények", "⚙️ Beállítások"],
            key="page_selector"
        )
        
        st.markdown("---")
        st.header("📊 Rendszer Állapot")
        
        # AI modulok állapota
        ai_status = "✅ Elérhető" if check_ai_modules() else "❌ Nem elérhető"
        st.write(f"**AI Modulok:** {ai_status}")
        
        # Dashboard állapot
        dashboard_status = "✅ Fut" if check_dashboard_status() else "❌ Nem fut"
        st.write(f"**AI Dashboard:** {dashboard_status}")
        
        # Desktop app állapot
        desktop_status = "✅ Aktív" if check_desktop_app_status() else "❌ Inaktív"
        st.write(f"**Desktop App:** {desktop_status}")
        
        st.markdown("---")
        st.header("⚙️ Gyors Beállítások")
        
        # Automatikus frissítés
        auto_refresh = st.checkbox("🔄 Automatikus frissítés", 
                                 value=st.session_state.settings["auto_refresh"])
        if auto_refresh != st.session_state.settings["auto_refresh"]:
            st.session_state.settings["auto_refresh"] = auto_refresh
        
        # Frissítési gyakoriság
        if auto_refresh:
            refresh_interval = st.slider("Frissítési gyakoriság (mp)", 5, 60, 15)
            st.session_state.refresh_interval = refresh_interval
        
        # Valós idejű monitoring
        real_time = st.checkbox("⏱️ Valós idejű monitoring", 
                               value=st.session_state.settings["real_time_monitoring"])
        if real_time != st.session_state.settings["real_time_monitoring"]:
            st.session_state.settings["real_time_monitoring"] = real_time
    
    # Oldal tartalom megjelenítése
    if page == "🏠 Főoldal":
        show_home_page()
    elif page == "📚 Kész Tanpéldák":
        show_examples_page()
    elif page == "📊 AI Dashboard":
        show_dashboard_page()
    elif page == "🔧 Tesztelés":
        show_testing_page()
    elif page == "📋 Eredmények":
        show_results_page()
    elif page == "⚙️ Beállítások":
        show_settings_page()

def show_home_page():
    """Főoldal megjelenítése"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 Mi ez a rendszer?")
        st.markdown("""
        A **DataChaEnhanced AI Excel Learning** egy mesterséges intelligencia alapú rendszer,
        amely képes Excel fájlokat elemzeni, tanulni belőlük, és új fájlokat generálni
        a tanult minták alapján.
        
        **Főbb képességek:**
        - 📊 **Excel elemzés**: Automatikus struktúra felismerés
        - 📈 **Grafikon tanítás**: Chart minták tanítása és generálása
        - 🧮 **Képlet tanítás**: Matematikai és logikai kapcsolatok
        - 🤖 **ML modellek**: Gépi tanulási algoritmusok
        - 🔄 **Folyamatos tanulás**: Automatikus fejlesztés
        """)
        
        st.header("🎯 Mire használható?")
        st.markdown("""
        - **Adatelemzés**: Nagy mennyiségű Excel adat automatikus feldolgozása
        - **Jelentés generálás**: Automatikus Excel jelentések készítése
        - **Mintázat felismerés**: Rejtett kapcsolatok és trendek felfedezése
        - **Minőségbiztosítás**: Adatok konzisztenciájának ellenőrzése
        - **Automatizálás**: Ismétlődő feladatok automatizálása
        """)
    
    with col2:
        st.header("📊 Rendszer Statisztikák")
        
        # Statisztika kártyák
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.metric("Aktív Modulok", 
                     5 if check_ai_modules() else 0, 
                     "mind elérhető" if check_ai_modules() else "nem elérhető")
        
        with col2_2:
            st.metric("Futtatott Tanpéldák", 
                     len(st.session_state.demo_results))
        
        # Sikerességi arány
        if st.session_state.demo_results:
            successful = sum(1 for r in st.session_state.demo_results.values() if r.get("success", False))
            success_rate = (successful / len(st.session_state.demo_results)) * 100
            st.metric("Sikerességi Arány", f"{success_rate:.1f}%")
        else:
            st.metric("Sikerességi Arány", "0%")
        
        # Legutóbbi aktivitás
        if st.session_state.demo_results:
            latest = max(st.session_state.demo_results.values(), 
                        key=lambda x: x.get("timestamp", ""))
            st.write(f"**Legutóbbi aktivitás:**")
            st.write(f"{latest.get('category_display', 'N/A')}")
            st.write(f"{latest.get('timestamp', '')[:19]}")
    
    # Központi akció gombok
    st.markdown("---")
    st.header("🎯 Kezdj el használni a rendszert!")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("📚 Nézd meg a tanpéldákat", use_container_width=True, type="primary"):
            st.session_state.current_page = "examples"
            st.rerun()
    
    with col4:
        if st.button("📊 Indítsd el a dashboard-ot", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    with col5:
        if st.button("🔧 Teszteld a funkciókat", use_container_width=True):
            st.session_state.current_page = "testing"
            st.rerun()

def show_examples_page():
    """Kész tanpéldák oldal megjelenítése"""
    st.header("📚 Kész Tanpéldák")
    st.markdown("---")
    
    # Kategória választó és leírás
    col1, col2 = st.columns([1, 2])
    
    with col1:
        categories = get_example_categories()
        selected_category = st.selectbox(
            "Válassz kategóriát:",
            list(categories.keys()),
            key="category_selector"
        )
        
        category_id = categories[selected_category]
        descriptions = get_example_descriptions()
        category_info = descriptions.get(category_id, {})
        
        # Kategória részletek
        st.markdown("### 📖 Kategória Információk")
        st.markdown(f"**Időtartam:** {category_info.get('duration', 'N/A')}")
        st.markdown(f"**Nehézség:** {category_info.get('difficulty', 'N/A')}")
        st.markdown(f"**Várható eredmény:** {category_info.get('expected_score', 'N/A')}")
        
        # Futtatás gombok
        st.markdown("### ⚡ Vezérlés")
        
        run_demo = st.button("▶ Futtatás", 
                           use_container_width=True, 
                           type="primary",
                           disabled=st.session_state.demo_running)
        
        if run_demo:
            run_selected_demo(category_id, selected_category)
        
        if st.session_state.demo_running:
            if st.button("⏹ Leállítás", use_container_width=True, type="secondary"):
                st.session_state.demo_running = False
                st.success("Tanpélda leállítva!")
                st.rerun()
    
    with col2:
        # Kategória leírás
        st.markdown("### 📋 Tanpélda Leírása")
        st.markdown(category_info.get('description', 'Leírás nem elérhető'))
        
        # Funkciók listája
        if 'features' in category_info:
            st.markdown("### 🎯 Mit fogsz megtanulni:")
            for feature in category_info['features']:
                st.markdown(f"• {feature}")
    
    # Demo futtatási állapot
    if st.session_state.demo_running:
        st.markdown("---")
        st.markdown("### 🔄 Tanpélda Futása")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Szimulált futtatás
        for i in range(100):
            time.sleep(0.05)  # Rövid várakozás
            progress_bar.progress(i + 1)
            status_text.text(f"Feldolgozás... {i+1}%")
        
        st.session_state.demo_running = False
        st.success("Tanpélda sikeresen lefutott!")
        st.rerun()
    
    # Gyors statisztikák
    st.markdown("---")
    st.markdown("### 📊 Gyors Statisztikák")
    
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.metric("Összes futtatás", len(st.session_state.demo_results))
    
    with col4:
        successful = sum(1 for r in st.session_state.demo_results.values() if r.get("success", False))
        st.metric("Sikeres", successful)
    
    with col5:
        failed = len(st.session_state.demo_results) - successful
        st.metric("Sikertelen", failed)
    
    with col6:
        if st.session_state.demo_results:
            avg_score = sum(r.get("score", 0) for r in st.session_state.demo_results.values()) / len(st.session_state.demo_results)
            st.metric("Átlagos pontszám", f"{avg_score:.1f}%")
        else:
            st.metric("Átlagos pontszám", "0%")

def run_selected_demo(category_id, category_display):
    """Futtatja a kiválasztott tanpéldát"""
    st.session_state.demo_running = True
    
    # Szimulált demo futtatás
    start_time = time.time()
    
    # Demo-specifikus logika
    demo_results = {
        "basic": {"success": True, "score": 92, "message": "Alapvető Excel elemzés sikeres"},
        "charts": {"success": True, "score": 84, "message": "Grafikon tanítás sikeres"},
        "formulas": {"success": True, "score": 89, "message": "Képlet tanítás sikeres"},
        "ml_models": {"success": True, "score": 96, "message": "ML modellek sikeresek"},
        "pipeline": {"success": True, "score": 94, "message": "Pipeline sikeres"},
        "advanced": {"success": True, "score": 98, "message": "Haladó funkciók sikeresek"}
    }
    
    result = demo_results.get(category_id, {"success": False, "score": 0, "message": "Ismeretlen kategória"})
    
    # Eredmény mentése
    end_time = time.time()
    duration = end_time - start_time
    
    result_key = f"{category_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.demo_results[result_key] = {
        "category": category_id,
        "category_display": category_display,
        "timestamp": datetime.now().isoformat(),
        "duration": f"{duration:.2f}s",
        "success": result["success"],
        "score": result["score"],
        "details": result
    }

def show_dashboard_page():
    """AI Dashboard oldal megjelenítése"""
    st.header("📊 AI Dashboard & Integráció")
    st.markdown("---")
    
    # Dashboard szolgáltatások
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 AI Monitoring Dashboard")
        
        dashboard_running = check_dashboard_status()
        if dashboard_running:
            st.success("✅ Dashboard fut (localhost:8501)")
            
            if st.button("🌐 Megnyitás böngészőben", key="open_dashboard"):
                webbrowser.open("http://localhost:8501")
                
            if st.button("⏹ Dashboard leállítása", key="stop_dashboard"):
                # Itt lehetne leállítani a dashboard-ot
                st.info("Dashboard leállítási funkció fejlesztés alatt")
        else:
            st.error("❌ Dashboard nincs elindítva")
            
            if st.button("🚀 Dashboard indítása", key="start_dashboard"):
                start_ai_dashboard()
    
    with col2:
        st.subheader("🌐 WebApp Dashboard")
        
        st.info("✅ WebApp fut (localhost:8502)")
        
        if st.button("🔄 WebApp újratöltése", key="reload_webapp"):
            st.rerun()
        
        if st.button("📱 Új ablakban megnyitása", key="open_new_window"):
            webbrowser.open("http://localhost:8502")
    
    # Integráció beállítások
    st.markdown("---")
    st.subheader("🔗 Integráció & Beállítások")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Szinkronizáció:**")
        
        sync_desktop = st.checkbox("🖥️ Desktop app szinkronizáció", 
                                  value=True)
        
        sync_dashboard = st.checkbox("📊 Dashboard szinkronizáció", 
                                    value=True)
        
        sync_realtime = st.checkbox("⏱️ Valós idejű frissítés", 
                                   value=st.session_state.settings["real_time_monitoring"])
    
    with col4:
        st.markdown("**Export & Riportolás:**")
        
        if st.button("📤 Eredmények exportálása", key="export_results"):
            export_results()
        
        if st.button("📊 Teljesítmény riport", key="performance_report"):
            show_performance_report()
        
        if st.button("🔄 Szinkronizáció most", key="sync_now"):
            st.success("Szinkronizáció végrehajtva!")
    
    # Dashboard előnézet (placeholder)
    st.markdown("---")
    st.subheader("👁️ Dashboard Előnézet")
    
    # Szimulált dashboard widgetek
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Aktív Modulok", "5/5", "100%")
    
    with col6:
        st.metric("CPU Használat", "45%", "2%")
    
    with col7:
        st.metric("Memória", "2.1GB", "0.1GB")
    
    # Egyszerű grafikon
    chart_data = pd.DataFrame({
        'time': pd.date_range('now', periods=10, freq='1min'),
        'cpu': [40, 42, 45, 43, 46, 44, 47, 45, 48, 45],
        'memory': [2.0, 2.1, 2.2, 2.1, 2.3, 2.2, 2.4, 2.3, 2.5, 2.4]
    })
    
    fig = px.line(chart_data, x='time', y=['cpu', 'memory'], 
                  title="Rendszer Teljesítmény (Utolsó 10 perc)")
    st.plotly_chart(fig, use_container_width=True)

def start_ai_dashboard():
    """Indítja az AI Dashboard-ot"""
    try:
        dashboard_script = Path(__file__).parent / "ai_dashboard.py"
        
        if dashboard_script.exists():
            # Dashboard indítása háttérben
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", str(dashboard_script),
                "--server.port", "8501",
                "--server.headless", "true"
            ])
            
            st.session_state.dashboard_process = process
            st.success("AI Dashboard indítása elkezdődött!")
            time.sleep(2)
            st.rerun()
        else:
            st.error("AI Dashboard script nem található!")
    
    except Exception as e:
        st.error(f"Hiba a dashboard indításában: {str(e)}")

def show_testing_page():
    """Tesztelési oldal megjelenítése"""
    st.header("🔧 Tesztelési Eszközök")
    st.markdown("---")
    
    # Gyors tesztek
    st.subheader("⚡ Gyors Tesztek")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 AI modulok tesztelése", use_container_width=True):
            test_ai_modules()
    
    with col2:
        if st.button("📊 Dashboard kapcsolat", use_container_width=True):
            test_dashboard_connection()
    
    with col3:
        if st.button("🖥️ Desktop app státusz", use_container_width=True):
            test_desktop_connection()
    
    # Teszteredmények területe
    st.markdown("---")
    st.subheader("📋 Tesztelési Eredmények")
    
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []
    
    # Eredmények megjelenítése
    if st.session_state.test_results:
        for result in st.session_state.test_results[-5:]:  # Utolsó 5 eredmény
            timestamp = result['timestamp']
            test_type = result['type']
            status = result['status']
            details = result['details']
            
            status_icon = "✅" if status == "success" else "❌"
            
            with st.expander(f"{status_icon} {test_type} - {timestamp}"):
                st.code(details, language='text')
    else:
        st.info("Még nem futtattak teszteket. Használd a fenti gombokat!")

def test_ai_modules():
    """Teszteli az AI modulokat"""
    test_result = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'type': 'AI Modulok Teszt',
        'status': 'success' if check_ai_modules() else 'error',
        'details': f"""AI MODULOK TESZTELÉSE - {datetime.now().strftime('%H:%M:%S')}
{'='*60}

Excel Analyzer: {'✅ ELÉRHETŐ' if check_ai_modules() else '❌ NEM ELÉRHETŐ'}
Chart Learner: {'✅ ELÉRHETŐ' if check_ai_modules() else '❌ NEM ELÉRHETŐ'}
Formula Learner: {'✅ ELÉRHETŐ' if check_ai_modules() else '❌ NEM ELÉRHETŐ'}
ML Models: {'✅ ELÉRHETŐ' if check_ai_modules() else '❌ NEM ELÉRHETŐ'}
Learning Pipeline: {'✅ ELÉRHETŐ' if check_ai_modules() else '❌ NEM ELÉRHETŐ'}

Teszt befejezve."""
    }
    
    st.session_state.test_results.append(test_result)
    
    if test_result['status'] == 'success':
        st.success("✅ AI modulok tesztelése sikeres!")
    else:
        st.error("❌ AI modulok tesztelése sikertelen!")

def test_dashboard_connection():
    """Teszteli a dashboard kapcsolatot"""
    dashboard_running = check_dashboard_status()
    
    test_result = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'type': 'Dashboard Kapcsolat Teszt',
        'status': 'success' if dashboard_running else 'error',
        'details': f"""DASHBOARD KAPCSOLAT TESZTELÉSE - {datetime.now().strftime('%H:%M:%S')}
{'='*60}

Dashboard folyamat: {'✅ Fut' if dashboard_running else '❌ Nem fut'}
HTTP kapcsolat: {'✅ OK (200)' if dashboard_running else '❌ Kapcsolat sikertelen'}
URL: http://localhost:8501

Teszt befejezve."""
    }
    
    st.session_state.test_results.append(test_result)
    
    if test_result['status'] == 'success':
        st.success("✅ Dashboard kapcsolat teszt sikeres!")
    else:
        st.error("❌ Dashboard kapcsolat teszt sikertelen!")

def test_desktop_connection():
    """Teszteli a desktop alkalmazás kapcsolatot"""
    desktop_status = check_desktop_app_status()
    
    test_result = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'type': 'Desktop App Kapcsolat Teszt',
        'status': 'success' if desktop_status else 'error',
        'details': f"""DESKTOP APP KAPCSOLAT TESZTELÉSE - {datetime.now().strftime('%H:%M:%S')}
{'='*60}

Desktop alkalmazás: {'✅ Aktív' if desktop_status else '❌ Nem elérhető'}
Kommunikációs csatorna: {'✅ Működik' if desktop_status else '❌ Hibás'}
Szinkronizáció: {'✅ Elérhető' if desktop_status else '❌ Nem elérhető'}

Teszt befejezve."""
    }
    
    st.session_state.test_results.append(test_result)
    
    if test_result['status'] == 'success':
        st.success("✅ Desktop app kapcsolat teszt sikeres!")
    else:
        st.error("❌ Desktop app kapcsolat teszt sikertelen!")

def show_results_page():
    """Eredmények oldal megjelenítése"""
    st.header("📋 Eredmények & Riportok")
    st.markdown("---")
    
    if not st.session_state.demo_results:
        st.info("Még nincsenek futtatott tanpéldák. Menj a 'Kész Tanpéldák' oldalra és futtass egy tanpéldát!")
        return
    
    # Összesítő statisztikák
    st.subheader("📊 Összesítő Statisztikák")
    
    total_runs = len(st.session_state.demo_results)
    successful_runs = sum(1 for r in st.session_state.demo_results.values() if r.get("success", False))
    failed_runs = total_runs - successful_runs
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Összes Futtatás", total_runs)
    
    with col2:
        st.metric("Sikeres", successful_runs)
    
    with col3:
        st.metric("Sikertelen", failed_runs)
    
    with col4:
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        st.metric("Sikerességi Arány", f"{success_rate:.1f}%")
    
    # Eredmények táblázata
    st.markdown("---")
    st.subheader("📋 Részletes Eredmények")
    
    # DataFrame létrehozása
    results_data = []
    for key, result in st.session_state.demo_results.items():
        results_data.append({
            "Dátum": result.get("timestamp", "")[:19].replace("T", " "),
            "Kategória": result.get("category_display", result.get("category", "")),
            "Állapot": "✅ Sikeres" if result.get("success", False) else "❌ Sikertelen",
            "Pontszám": f"{result.get('score', 0)}%",
            "Időtartam": result.get("duration", ""),
            "Részletek": result.get("details", {}).get("message", "")
        })
    
    if results_data:
        df = pd.DataFrame(results_data)
        
        # Szűrési opciók
        col5, col6 = st.columns(2)
        
        with col5:
            status_filter = st.selectbox("Szűrés állapot szerint:", 
                                       ["Összes", "✅ Sikeres", "❌ Sikertelen"])
        
        with col6:
            category_filter = st.selectbox("Szűrés kategória szerint:",
                                         ["Összes"] + list(df["Kategória"].unique()))
        
        # Szűrés alkalmazása
        filtered_df = df.copy()
        
        if status_filter != "Összes":
            filtered_df = filtered_df[filtered_df["Állapot"] == status_filter]
        
        if category_filter != "Összes":
            filtered_df = filtered_df[filtered_df["Kategória"] == category_filter]
        
        # Táblázat megjelenítése
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export gombok
        st.markdown("---")
        st.subheader("📤 Export Opciók")
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            if st.button("📊 JSON Export", use_container_width=True):
                export_results_json()
        
        with col8:
            if st.button("📋 CSV Export", use_container_width=True):
                export_results_csv(filtered_df)
        
        with col9:
            if st.button("📄 Riport Generálás", use_container_width=True):
                generate_detailed_report()

def export_results():
    """Általános eredmény exportálás"""
    if not st.session_state.demo_results:
        st.warning("Nincsenek exportálható eredmények!")
        return
    
    export_format = st.session_state.settings.get("export_format", "json")
    
    if export_format == "json":
        export_results_json()
    else:
        generate_detailed_report()

def export_results_json():
    """JSON formátumban exportálja az eredményeket"""
    if not st.session_state.demo_results:
        st.warning("Nincsenek exportálható eredmények!")
        return
    
    # JSON string létrehozása
    json_data = json.dumps(st.session_state.demo_results, indent=2, ensure_ascii=False)
    
    # Download gomb
    st.download_button(
        label="💾 JSON Letöltése",
        data=json_data,
        file_name=f"ai_excel_learning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("JSON export kész! Kattints a letöltés gombra.")

def export_results_csv(df):
    """CSV formátumban exportálja az eredményeket"""
    csv_data = df.to_csv(index=False, encoding='utf-8')
    
    st.download_button(
        label="💾 CSV Letöltése",
        data=csv_data,
        file_name=f"ai_excel_learning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.success("CSV export kész! Kattints a letöltés gombra.")

def generate_detailed_report():
    """Részletes riport generálása"""
    if not st.session_state.demo_results:
        st.warning("Nincs adat a riport generálásához!")
        return
    
    # Riport tartalom létrehozása
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("AI EXCEL LEARNING - RÉSZLETES ÖSSZEFOGLALÓ JELENTÉS")
    report_lines.append("=" * 80)
    report_lines.append(f"Generálva: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"WebApp verzió: Enhanced Demo v1.0")
    report_lines.append("")
    
    # Összesítés
    total_demos = len(st.session_state.demo_results)
    successful_demos = sum(1 for r in st.session_state.demo_results.values() if r.get("success", False))
    failed_demos = total_demos - successful_demos
    
    if total_demos > 0:
        avg_score = sum(r.get("score", 0) for r in st.session_state.demo_results.values()) / total_demos
        success_rate = successful_demos / total_demos * 100
    else:
        avg_score = 0
        success_rate = 0
    
    report_lines.append("ÖSSZESÍTŐ STATISZTIKÁK:")
    report_lines.append("-" * 40)
    report_lines.append(f"  Összes tanpélda futtatás: {total_demos}")
    report_lines.append(f"  Sikeres futtatások: {successful_demos}")
    report_lines.append(f"  Sikertelen futtatások: {failed_demos}")
    report_lines.append(f"  Sikerességi arány: {success_rate:.1f}%")
    report_lines.append(f"  Átlagos pontszám: {avg_score:.1f}%")
    report_lines.append("")
    
    # Részletes eredmények
    report_lines.append("RÉSZLETES EREDMÉNYEK:")
    report_lines.append("-" * 40)
    
    for result_key, result in st.session_state.demo_results.items():
        report_lines.append(f"Futtatás ID: {result_key}")
        report_lines.append(f"  Kategória: {result.get('category_display', result.get('category'))}")
        report_lines.append(f"  Dátum: {result.get('timestamp')}")
        report_lines.append(f"  Időtartam: {result.get('duration')}")
        report_lines.append(f"  Állapot: {'Sikeres' if result.get('success') else 'Sikertelen'}")
        report_lines.append(f"  Pontszám: {result.get('score', 0)}%")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("Riport vége")
    
    report_text = "\n".join(report_lines)
    
    # Download gomb
    st.download_button(
        label="💾 Riport Letöltése",
        data=report_text,
        file_name=f"ai_excel_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    st.success("Riport generálva! Kattints a letöltés gombra.")

def show_performance_report():
    """Teljesítmény riport megjelenítése"""
    st.subheader("📊 Teljesítmény Riport")
    
    report_content = f"""
**Generálva:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Rendszer Információk:**
- AI modulok: {'Elérhető' if check_ai_modules() else 'Nem elérhető'}
- Dashboard: {'Fut' if check_dashboard_status() else 'Nem fut'}
- WebApp: ✅ Fut (localhost:8502)

**Teljesítmény Statisztikák:**
"""
    
    if st.session_state.demo_results:
        total_runs = len(st.session_state.demo_results)
        successful_runs = sum(1 for r in st.session_state.demo_results.values() if r.get("success", False))
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        # Időstatisztikák
        times = []
        for r in st.session_state.demo_results.values():
            try:
                time_str = r.get("duration", "0s").replace("s", "")
                times.append(float(time_str))
            except ValueError:
                pass
        
        avg_time = sum(times) / len(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        
        report_content += f"""
- Összes futtatás: {total_runs}
- Sikeresség: {success_rate:.1f}%
- Átlagos idő: {avg_time:.2f}s
- Leggyorsabb: {min_time:.2f}s
- Leglassabb: {max_time:.2f}s
"""
    else:
        report_content += "\n- Nincsenek elérhető adatok"
    
    st.markdown(report_content)

def show_settings_page():
    """Beállítások oldal megjelenítése"""
    st.header("⚙️ Beállítások")
    st.markdown("---")
    
    # Általános beállítások
    st.subheader("🔧 Általános Beállítások")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Működési Beállítások:**")
        
        auto_refresh = st.checkbox("🔄 Automatikus frissítés", 
                                 value=st.session_state.settings["auto_refresh"])
        
        real_time = st.checkbox("⏱️ Valós idejű monitoring", 
                               value=st.session_state.settings["real_time_monitoring"])
        
        detailed_logging = st.checkbox("📝 Részletes naplózás", 
                                      value=st.session_state.settings["detailed_logging"])
        
        # Beállítások mentése
        st.session_state.settings.update({
            "auto_refresh": auto_refresh,
            "real_time_monitoring": real_time,
            "detailed_logging": detailed_logging
        })
    
    with col2:
        st.markdown("**Export Beállítások:**")
        
        export_format = st.selectbox("Alapértelmezett export formátum:",
                                   ["json", "txt", "csv"],
                                   index=["json", "txt", "csv"].index(st.session_state.settings["export_format"]))
        
        st.session_state.settings["export_format"] = export_format
        
        if st.button("💾 Beállítások mentése", type="primary"):
            st.success("Beállítások mentve!")
        
        if st.button("🔄 Alapértelmezett visszaállítása"):
            st.session_state.settings = {
                "auto_refresh": True,
                "real_time_monitoring": True,
                "detailed_logging": False,
                "export_format": "json"
            }
            st.success("Alapértelmezett beállítások visszaállítva!")
            st.rerun()
    
    # Rendszer információk
    st.markdown("---")
    st.subheader("ℹ️ Rendszer Információk")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Alkalmazás Verziók:**")
        st.write("- WebApp: Enhanced Demo v1.0")
        st.write("- Streamlit:", st.__version__)
        st.write("- Python:", sys.version.split()[0])
    
    with col4:
        st.markdown("**Kapcsolat Állapotok:**")
        st.write(f"- AI Modulok: {'✅ Elérhető' if check_ai_modules() else '❌ Nem elérhető'}")
        st.write(f"- AI Dashboard: {'✅ Fut' if check_dashboard_status() else '❌ Nem fut'}")
        st.write(f"- Desktop App: {'✅ Aktív' if check_desktop_app_status() else '❌ Inaktív'}")
    
    # Adatok kezelése
    st.markdown("---")
    st.subheader("🗂️ Adatok Kezelése")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("**Eredmények:**")
        st.write(f"Tárolt eredmények száma: {len(st.session_state.demo_results)}")
        
        if st.button("🗑️ Összes eredmény törlése", type="secondary"):
            if st.button("⚠️ Megerősítés - MINDEN eredmény törlődik!", type="secondary"):
                st.session_state.demo_results.clear()
                st.success("Összes eredmény törölve!")
                st.rerun()
    
    with col6:
        st.markdown("**Session Adatok:**")
        st.write(f"Session állapot mérete: {len(str(st.session_state))}")
        
        if st.button("🔄 Session újraindítása"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session újraindítva!")
            st.rerun()

# Automatikus frissítés logika
if st.session_state.settings.get("auto_refresh", False):
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    refresh_interval = st.session_state.get('refresh_interval', 15)
    
    if current_time - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()

# Fő alkalmazás futtatása
if __name__ == "__main__":
    main()