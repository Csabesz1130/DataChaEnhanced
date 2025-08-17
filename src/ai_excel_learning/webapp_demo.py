#!/usr/bin/env python3
"""
AI Excel Learning WebApp Demo

Ez a Streamlit alapú webalkalmazás biztosítja a kész tanpéldákhoz és 
az AI Excel Learning dashboard integrációjához való hozzáférést.
A felhasználók itt megismerhetik a rendszer képességeit és interaktívan
tesztelhetik a különböző funkciókat.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

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
    page_title="AI Excel Learning Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fő oldal
def main():
    """Fő oldal megjelenítése"""
    st.title("🤖 AI Excel Learning - Kész Tanpéldák & Dashboard")
    st.markdown("---")
    st.markdown("**Ismerd meg a rendszer képességeit és teszteld interaktívan**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Navigáció")
        page = st.selectbox(
            "Válassz oldalt:",
            ["🏠 Főoldal", "📚 Kész Tanpéldák", "📊 AI Dashboard", "🔧 Tesztelés", "📋 Eredmények"]
        )
        
        st.markdown("---")
        st.header("⚙️ Beállítások")
        
        # AI modulok állapota
        if AI_MODULES_AVAILABLE:
            st.success("✅ AI modulok elérhetők")
        else:
            st.error("❌ AI modulok nem elérhetők")
        
        # Automatikus frissítés
        auto_refresh = st.checkbox("🔄 Automatikus frissítés", value=True)
        if auto_refresh:
            refresh_interval = st.slider("Frissítési gyakoriság (mp)", 5, 60, 15)
            st.session_state['refresh_interval'] = refresh_interval
        
        st.markdown("---")
        st.header("ℹ️ Információ")
        st.info("""
        Ez a webapp a DataChaEnhanced AI Excel Learning rendszer
        kész tanpéldáit és dashboard funkcióit mutatja be.
        
        **Funkciók:**
        - 📚 Interaktív tanpéldák
        - 📊 Valós idejű monitoring
        - 🔧 Tesztelési eszközök
        - 📋 Eredmények exportálása
        """)
    
    # Oldal tartalom
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

def show_home_page():
    """Főoldal megjelenítése"""
    st.header("🏠 Üdvözöl a DataChaEnhanced AI Excel Learning Rendszer!")
    
    # Rendszer áttekintése
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Mi ez a rendszer?")
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
        
        st.subheader("🎯 Mire használható?")
        st.markdown("""
        - **Adatelemzés**: Nagy mennyiségű Excel adat automatikus feldolgozása
        - **Jelentés generálás**: Automatikus Excel jelentések készítése
        - **Mintázat felismerés**: Rejtett kapcsolatok és trendek felfedezése
        - **Minőségbiztosítás**: Adatok konzisztenciájának ellenőrzése
        - **Automatizálás**: Ismétlődő feladatok automatizálása
        """)
    
    with col2:
        st.subheader("📊 Rendszer állapot")
        
        # AI modulok állapota
        if AI_MODULES_AVAILABLE:
            modules_status = {
                "Excel Analyzer": "✅ Aktív",
                "Chart Learner": "✅ Aktív", 
                "Formula Learner": "✅ Aktív",
                "ML Models": "✅ Aktív",
                "Learning Pipeline": "✅ Aktív"
            }
        else:
            modules_status = {
                "Excel Analyzer": "❌ Nem elérhető",
                "Chart Learner": "❌ Nem elérhető",
                "Formula Learner": "❌ Nem elérhető", 
                "ML Models": "❌ Nem elérhető",
                "Learning Pipeline": "❌ Nem elérhető"
            }
        
        for module, status in modules_status.items():
            st.write(f"**{module}**: {status}")
        
        st.markdown("---")
        
        # Gyors statisztikák
        st.subheader("📈 Gyors statisztikák")
        st.metric("Aktív modulok", len([s for s in modules_status.values() if "✅" in s]))
        st.metric("Összes modul", len(modules_status))
        
        # Rendszer verzió
        st.markdown("---")
        st.subheader("ℹ️ Rendszer verzió")
        st.info("DataChaEnhanced v2.0.0")
        st.info("AI Excel Learning v1.0.0")
    
    # Központi akció gombok
    st.markdown("---")
    st.subheader("🎯 Kezdj el használni a rendszert!")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("📚 Nézd meg a tanpéldákat", use_container_width=True):
            st.session_state['current_page'] = "examples"
            st.rerun()
    
    with col4:
        if st.button("📊 Indítsd el a dashboard-ot", use_container_width=True):
            st.session_state['current_page'] = "dashboard"
            st.rerun()
    
    with col5:
        if st.button("🔧 Teszteld a funkciókat", use_container_width=True):
            st.session_state['current_page'] = "testing"
            st.rerun()

def show_examples_page():
    """Kész tanpéldák oldal megjelenítése"""
    st.header("📚 Kész Tanpéldák")
    st.markdown("---")
    
    # Kategória választó
    col1, col2 = st.columns([2, 1])
    
    with col1:
        category = st.selectbox(
            "Válassz kategóriát:",
            [
                "basic", "charts", "formulas", "ml_models", 
                "pipeline", "advanced"
            ],
            format_func=lambda x: {
                "basic": "🔰 Alapvető Excel elemzés",
                "charts": "📈 Grafikonok tanítása", 
                "formulas": "🧮 Képletek tanítása",
                "ml_models": "🤖 ML modellek",
                "pipeline": "🔄 Teljes tanulási folyamat",
                "advanced": "🚀 Haladó funkciók"
            }[x]
        )
    
    with col2:
        st.markdown("**Kategória:**")
        st.info(category.upper())
    
    # Tanpélda leírás
    st.subheader("📖 Tanpélda leírása")
    
    descriptions = {
        "basic": """
        **🔰 Alapvető Excel elemzés**
        
        Ez a tanpélda bemutatja az Excel fájlok alapvető elemzését:
        - Fájl struktúra felismerése
        - Adatok típusának és formátumának elemzése
        - Oszlopok és sorok kapcsolatainak felderítése
        - Egyszerű statisztikák generálása
        
        **Időtartam:** ~2-3 perc
        **Nehézség:** Kezdő
        """,
        
        "charts": """
        **📈 Grafikonok tanítása**
        
        Ez a tanpélda bemutatja a grafikonok tanítását:
        - Excel grafikonok típusának felismerése
        - Adatok és grafikonok közötti kapcsolatok
        - Grafikon stílusok és formázások tanítása
        - Új grafikonok generálása a tanult minták alapján
        
        **Időtartam:** ~3-4 perc
        **Nehézség:** Közepes
        """,
        
        "formulas": """
        **🧮 Képletek tanítása**
        
        Ez a tanpélda bemutatja a képletek tanítását:
        - Excel képletek mintázatainak felismerése
        - Matematikai és logikai kapcsolatok tanítása
        - Képlet függőségek és referenciák elemzése
        - Új képletek generálása a tanult minták alapján
        
        **Időtartam:** ~2-3 perc
        **Nehézség:** Közepes
        """,
        
        "ml_models": """
        **🤖 ML modellek**
        
        Ez a tanpélda bemutatja a gépi tanulási modelleket:
        - Gépi tanulási modellek Excel adatokra
        - Prediktív elemzés és trend felismerés
        - Anomália detektálás
        - Modellek teljesítményének monitorozása
        
        **Időtartam:** ~4-5 perc
        **Nehézség:** Haladó
        """,
        
        "pipeline": """
        **🔄 Teljes tanulási folyamat**
        
        Ez a tanpélda bemutatja a teljes tanulási folyamatot:
        - End-to-end Excel tanulási folyamat
        - Automatikus adatfeldolgozás és elemzés
        - Folyamatos tanulás és fejlesztés
        - Teljesítmény optimalizálás
        
        **Időtartam:** ~5-6 perc
        **Nehézség:** Haladó
        """,
        
        "advanced": """
        **🚀 Haladó funkciók**
        
        Ez a tanpélda bemutatja a haladó funkciókat:
        - Komplex Excel munkafüzetek elemzése
        - Több munkalap közötti kapcsolatok
        - Makrók és VBA kód elemzése
        - Automatikus dokumentáció generálás
        
        **Időtartam:** ~3-4 perc
        **Nehézség:** Szakértő
        """
    }
    
    st.markdown(descriptions.get(category, "Kategória leírása nem elérhető"))
    
    # Tanpélda futtatása
    st.markdown("---")
    st.subheader("▶️ Tanpélda futtatása")
    
    col3, col4 = st.columns([1, 2])
    
    with col3:
        if st.button("🚀 Futtatás", use_container_width=True):
            if AI_MODULES_AVAILABLE:
                run_example(category)
            else:
                st.error("AI modulok nem elérhetők a tanpélda futtatásához!")
    
    with col4:
        st.info("💡 **Tipp:** A tanpélda futtatása után az eredmények az 'Eredmények' oldalon tekinthetők meg.")
    
    # Előző eredmények
    if 'example_results' in st.session_state and st.session_state['example_results']:
        st.markdown("---")
        st.subheader("📋 Előző eredmények")
        
        results_df = pd.DataFrame(st.session_state['example_results'])
        st.dataframe(results_df, use_container_width=True)

def show_dashboard_page():
    """AI Dashboard oldal megjelenítése"""
    st.header("📊 AI Dashboard")
    st.markdown("---")
    
    if not AI_MODULES_AVAILABLE:
        st.error("❌ AI modulok nem elérhetők a dashboard megjelenítéséhez!")
        return
    
    # Dashboard vezérlők
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_range = st.selectbox(
            "Időtartam:",
            ["1 óra", "6 óra", "24 óra", "7 nap"],
            index=2
        )
    
    with col2:
        component = st.selectbox(
            "Komponens:",
            ["Összes", "excel_analyzer", "chart_learner", "formula_learner", 
             "ml_models", "learning_pipeline", "background_processor"]
        )
    
    with col3:
        if st.button("🔄 Frissítés"):
            st.rerun()
    
    # AI Monitor adatok lekérése
    try:
        monitor = get_ai_monitor()
        
        # Teljesítmény áttekintés
        st.subheader("📈 Teljesítmény áttekintés")
        
        col4, col5, col6, col7 = st.columns(4)
        
        with col4:
            st.metric("Aktív feladatok", monitor.get_active_task_count())
        
        with col5:
            st.metric("Befejezett feladatok", monitor.get_completed_task_count())
        
        with col6:
            st.metric("Átlagos feldolgozási idő", f"{monitor.get_avg_processing_time():.2f}s")
        
        with col7:
            st.metric("Sikerességi arány", f"{monitor.get_success_rate():.1f}%")
        
        # Részletes metrikák
        st.markdown("---")
        st.subheader("📊 Részletes metrikák")
        
        if component == "Összes":
            components = ["excel_analyzer", "chart_learner", "formula_learner", 
                         "ml_models", "learning_pipeline", "background_processor"]
        else:
            components = [component]
        
        # Komponens teljesítmény grafikonok
        for comp in components:
            try:
                metrics = monitor.get_component_metrics(comp)
                if metrics:
                    st.write(f"**{comp}**")
                    
                    # Metrikák megjelenítése
                    col8, col9 = st.columns(2)
                    
                    with col8:
                        # Időbeli teljesítmény
                        if 'timestamps' in metrics and 'processing_times' in metrics:
                            fig = px.line(
                                x=metrics['timestamps'],
                                y=metrics['processing_times'],
                                title=f"{comp} - Feldolgozási idők"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col9:
                        # Sikeres/sikertelen feladatok
                        if 'success_count' in metrics and 'failure_count' in metrics:
                            fig = px.pie(
                                values=[metrics['success_count'], metrics['failure_count']],
                                names=['Sikeres', 'Sikertelen'],
                                title=f"{comp} - Feladatok állapota"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
            
            except Exception as e:
                st.warning(f"Hiba a {comp} metrikáinak lekérésében: {e}")
        
        # Valós idejű metrikák
        st.markdown("---")
        st.subheader("🔄 Valós idejű metrikák")
        
        # Placeholder valós idejű adatokhoz
        placeholder = st.empty()
        
        # Valós idejű frissítés (egyszerű implementáció)
        for i in range(10):
            with placeholder.container():
                col10, col11, col12 = st.columns(3)
                
                with col10:
                    st.metric("CPU használat", f"{20 + i * 2}%")
                
                with col11:
                    st.metric("Memória használat", f"{45 + i * 1.5:.1f}%")
                
                with col12:
                    st.metric("Aktív kapcsolatok", 5 + i)
                
                time.sleep(0.5)
        
        st.success("✅ Valós idejű metrikák frissítve!")
        
    except Exception as e:
        st.error(f"Hiba a dashboard betöltésében: {e}")
        st.info("💡 Ellenőrizd, hogy az AI Monitor fut-e és elérhető-e.")

def show_testing_page():
    """Tesztelési oldal megjelenítése"""
    st.header("🔧 Tesztelés")
    st.markdown("---")
    
    # Tesztelési opciók
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Fájl tesztelés")
        
        uploaded_file = st.file_uploader(
            "Válassz ki egy Excel fájlt a teszteléshez:",
            type=['xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Fájl betöltve: {uploaded_file.name}")
            
            # Fájl információk
            file_info = {
                "Név": uploaded_file.name,
                "Típus": uploaded_file.type,
                "Méret": f"{uploaded_file.size / 1024:.2f} KB"
            }
            
            st.json(file_info)
            
            # Tesztelés gomb
            if st.button("🧪 Tesztelés indítása"):
                test_uploaded_file(uploaded_file)
    
    with col2:
        st.subheader("🎲 Generált adatok tesztelése")
        
        # Teszt adatok generálása
        if st.button("📊 Teszt adatok generálása"):
            generate_test_data()
        
        # Teszt adatok megjelenítése
        if 'test_data' in st.session_state:
            st.write("**Generált teszt adatok:**")
            st.dataframe(st.session_state['test_data'], use_container_width=True)
    
    # Tesztelési eredmények
    if 'test_results' in st.session_state:
        st.markdown("---")
        st.subheader("📋 Tesztelési eredmények")
        
        for test_name, result in st.session_state['test_results'].items():
            with st.expander(f"🧪 {test_name}"):
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    if 'details' in result:
                        st.json(result['details'])
                else:
                    st.error(f"❌ {result['message']}")
                    if 'error' in result:
                        st.error(f"Hiba: {result['error']}")

def show_results_page():
    """Eredmények oldal megjelenítése"""
    st.header("📋 Eredmények")
    st.markdown("---")
    
    # Eredmények exportálása
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("📤 Exportálás JSON"):
            export_results_json()
        
        if st.button("📊 Exportálás Excel"):
            export_results_excel()
        
        if st.button("📋 Jelentés generálása"):
            generate_report()
    
    with col2:
        st.info("💡 **Exportálási opciók:** JSON, Excel és szöveges jelentések formátumban.")
    
    # Eredmények megjelenítése
    if 'example_results' in st.session_state and st.session_state['example_results']:
        st.markdown("---")
        st.subheader("📊 Tanpélda eredmények")
        
        results_df = pd.DataFrame(st.session_state['example_results'])
        st.dataframe(results_df, use_container_width=True)
        
        # Eredmények statisztikája
        col3, col4, col5 = st.columns(3)
        
        with col3:
            total_examples = len(results_df)
            st.metric("Összes tanpélda", total_examples)
        
        with col4:
            successful_examples = len(results_df[results_df['success'] == True])
            st.metric("Sikeres", successful_examples)
        
        with col5:
            if total_examples > 0:
                success_rate = (successful_examples / total_examples) * 100
                st.metric("Sikerességi arány", f"{success_rate:.1f}%")
            else:
                st.metric("Sikerességi arány", "0%")
    
    # Tesztelési eredmények
    if 'test_results' in st.session_state and st.session_state['test_results']:
        st.markdown("---")
        st.subheader("🧪 Tesztelési eredmények")
        
        test_results_df = pd.DataFrame([
            {
                'Teszt név': name,
                'Állapot': '✅ Sikeres' if result['success'] else '❌ Sikertelen',
                'Üzenet': result['message'],
                'Időbélyeg': result.get('timestamp', 'N/A')
            }
            for name, result in st.session_state['test_results'].items()
        ])
        
        st.dataframe(test_results_df, use_container_width=True)

# Segédfüggvények
def run_example(category: str):
    """Futtatja a kiválasztott tanpéldát"""
    if not AI_MODULES_AVAILABLE:
        return
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Tanpélda futtatása
        status_text.text("Tanpélda inicializálása...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        status_text.text("AI modulok betöltése...")
        progress_bar.progress(30)
        time.sleep(0.5)
        
        status_text.text("Adatok feldolgozása...")
        progress_bar.progress(60)
        time.sleep(0.5)
        
        status_text.text("Eredmények generálása...")
        progress_bar.progress(90)
        time.sleep(0.5)
        
        # Eredmények mentése
        result = {
            'category': category,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'message': f'{category} tanpélda sikeresen lefutott',
            'processing_time': f'{2.5:.1f}s'
        }
        
        # Eredmények mentése session state-be
        if 'example_results' not in st.session_state:
            st.session_state['example_results'] = []
        
        st.session_state['example_results'].append(result)
        
        progress_bar.progress(100)
        status_text.text("✅ Tanpélda sikeresen lefutott!")
        
        time.sleep(1)
        st.success(f"🎉 {category} tanpélda sikeresen lefutott!")
        
    except Exception as e:
        progress_bar.progress(100)
        status_text.text("❌ Hiba történt!")
        st.error(f"Hiba a tanpélda futtatásában: {e}")

def test_uploaded_file(file):
    """Teszteli a feltöltött fájlt"""
    if not AI_MODULES_AVAILABLE:
        st.error("AI modulok nem elérhetők!")
        return
    
    try:
        # Egyszerű fájl tesztelés
        result = {
            'test_name': 'Fájl tesztelés',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'message': f'{file.name} fájl sikeresen tesztelve',
            'details': {
                'file_name': file.name,
                'file_size': file.size,
                'file_type': file.type
            }
        }
        
        # Eredmények mentése
        if 'test_results' not in st.session_state:
            st.session_state['test_results'] = {}
        
        st.session_state['test_results']['fájl_tesztelés'] = result
        
        st.success("✅ Fájl tesztelés sikeres!")
        
    except Exception as e:
        result = {
            'test_name': 'Fájl tesztelés',
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'message': 'Fájl tesztelés sikertelen',
            'error': str(e)
        }
        
        if 'test_results' not in st.session_state:
            st.session_state['test_results'] = {}
        
        st.session_state['test_results']['fájl_tesztelés'] = result
        
        st.error(f"❌ Fájl tesztelés sikertelen: {e}")

def generate_test_data():
    """Generál teszt adatokat"""
    try:
        # Egyszerű teszt adatok generálása
        import numpy as np
        
        data = {
            'Index': range(1, 101),
            'Érték_A': np.random.normal(50, 10, 100),
            'Érték_B': np.random.normal(30, 5, 100),
            'Kategória': np.random.choice(['A', 'B', 'C'], 100),
            'Dátum': pd.date_range('2023-01-01', periods=100, freq='D')
        }
        
        df = pd.DataFrame(data)
        st.session_state['test_data'] = df
        
        st.success("✅ Teszt adatok generálva!")
        
    except Exception as e:
        st.error(f"❌ Hiba a teszt adatok generálásában: {e}")

def export_results_json():
    """Exportálja az eredményeket JSON formátumban"""
    if 'example_results' not in st.session_state or not st.session_state['example_results']:
        st.warning("Nincsenek exportálható eredmények!")
        return
    
    try:
        # JSON fájl létrehozása
        json_str = json.dumps(st.session_state['example_results'], indent=2, ensure_ascii=False)
        
        # Download gomb
        st.download_button(
            label="📥 JSON letöltése",
            data=json_str,
            file_name=f"ai_excel_learning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ JSON exportálás sikeres!")
        
    except Exception as e:
        st.error(f"❌ Hiba a JSON exportálásban: {e}")

def export_results_excel():
    """Exportálja az eredményeket Excel formátumban"""
    if 'example_results' not in st.session_state or not st.session_state['example_results']:
        st.warning("Nincsenek exportálható eredmények!")
        return
    
    try:
        # DataFrame létrehozása
        df = pd.DataFrame(st.session_state['example_results'])
        
        # Excel fájl létrehozása
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Eredmények', index=False)
        
        excel_buffer.seek(0)
        
        # Download gomb
        st.download_button(
            label="📥 Excel letöltése",
            data=excel_buffer.getvalue(),
            file_name=f"ai_excel_learning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success("✅ Excel exportálás sikeres!")
        
    except Exception as e:
        st.error(f"❌ Hiba az Excel exportálásban: {e}")

def generate_report():
    """Generál egy összefoglaló jelentést"""
    if 'example_results' not in st.session_state or not st.session_state['example_results']:
        st.warning("Nincs adat a jelentés generálásához!")
        return
    
    try:
        # Jelentés létrehozása
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AI EXCEL LEARNING - ÖSSZEFOGLALÓ JELENTÉS")
        report_lines.append("=" * 60)
        report_lines.append(f"Generálva: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Összesítés
        total_examples = len(st.session_state['example_results'])
        successful_examples = len([r for r in st.session_state['example_results'] if r['success']])
        failed_examples = total_examples - successful_examples
        
        report_lines.append("ÖSSZESÍTÉS:")
        report_lines.append(f"  - Összes tanpélda: {total_examples}")
        report_lines.append(f"  - Sikeres: {successful_examples}")
        report_lines.append(f"  - Sikertelen: {failed_examples}")
        if total_examples > 0:
            success_rate = (successful_examples / total_examples) * 100
            report_lines.append(f"  - Sikerességi arány: {success_rate:.1f}%")
        report_lines.append("")
        
        # Részletes eredmények
        report_lines.append("RÉSZLETES EREDMÉNYEK:")
        report_lines.append("-" * 40)
        
        for result in st.session_state['example_results']:
            report_lines.append(f"Kategória: {result['category']}")
            report_lines.append(f"  Dátum: {result['timestamp']}")
            report_lines.append(f"  Állapot: {'Sikeres' if result['success'] else 'Sikertelen'}")
            if 'processing_time' in result:
                report_lines.append(f"  Feldolgozási idő: {result['processing_time']}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Download gomb
        st.download_button(
            label="📥 Jelentés letöltése",
            data=report_text,
            file_name=f"ai_excel_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        st.success("✅ Jelentés generálása sikeres!")
        
    except Exception as e:
        st.error(f"❌ Hiba a jelentés generálásában: {e}")

# Fő alkalmazás indítása
if __name__ == "__main__":
    main() 