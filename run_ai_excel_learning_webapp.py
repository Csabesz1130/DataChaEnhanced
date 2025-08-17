#!/usr/bin/env python3
"""
AI Excel Learning WebApp Indító Script

Ez a script indítja az AI Excel Learning webapp-ot Streamlit segítségével.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Fő függvény a webapp indításához"""
    print("🤖 AI Excel Learning WebApp indítása...")
    
    # Ellenőrizzük a szükséges függőségeket
    try:
        import streamlit
        print("✅ Streamlit elérhető")
    except ImportError:
        print("❌ Streamlit nincs telepítve!")
        print("Telepítés: pip install streamlit")
        return
    
    try:
        import pandas
        print("✅ Pandas elérhető")
    except ImportError:
        print("❌ Pandas nincs telepítve!")
        print("Telepítés: pip install pandas")
        return
    
    try:
        import plotly
        print("✅ Plotly elérhető")
    except ImportError:
        print("❌ Plotly nincs telepítve!")
        print("Telepítés: pip install plotly")
        return
    
    # Webapp fájl elérési útja
    webapp_path = Path(__file__).parent / "src" / "ai_excel_learning" / "webapp_demo.py"
    
    if not webapp_path.exists():
        print(f"❌ Webapp fájl nem található: {webapp_path}")
        return
    
    print(f"✅ Webapp fájl található: {webapp_path}")
    
    # Webapp indítása
    print("🚀 Webapp indítása...")
    print("📱 A webapp a böngészőben fog megnyílni")
    print("🔄 A webapp leállításához nyomd meg a Ctrl+C billentyűkombinációt")
    print("-" * 50)
    
    try:
        # Streamlit webapp indítása
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(webapp_path),
            "--server.port", "8502",  # Külön port a fő dashboard-tól
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Webapp leállítva a felhasználó által")
    except Exception as e:
        print(f"❌ Hiba a webapp indításában: {e}")

if __name__ == "__main__":
    main() 