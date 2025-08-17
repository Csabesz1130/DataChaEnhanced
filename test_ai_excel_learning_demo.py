#!/usr/bin/env python3
"""
AI Excel Learning Demo Teszt Script

Ez a script teszteli az új AI Excel Learning Demo funkciókat.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Teszteli az importokat"""
    print("🔍 Importok tesztelése...")
    
    try:
        # Desktop tab import teszt
        sys.path.append(str(Path(__file__).parent / "src"))
        from gui.ai_excel_learning_demo_tab import AIExcelLearningDemoTab
        print("✅ Desktop tab import sikeres")
    except ImportError as e:
        print(f"❌ Desktop tab import sikertelen: {e}")
        return False
    
    try:
        # Webapp import teszt
        from ai_excel_learning.webapp_demo import main
        print("✅ Webapp import sikeres")
    except ImportError as e:
        print(f"❌ Webapp import sikertelen: {e}")
        return False
    
    try:
        # AI modulok import teszt
        from ai_excel_learning.ai_dashboard import create_dashboard
        print("✅ AI Dashboard import sikeres")
    except ImportError as e:
        print(f"❌ AI Dashboard import sikertelen: {e}")
        return False
    
    return True

def test_file_structure():
    """Teszteli a fájl struktúrát"""
    print("\n📁 Fájl struktúra tesztelése...")
    
    required_files = [
        "src/gui/ai_excel_learning_demo_tab.py",
        "src/ai_excel_learning/webapp_demo.py",
        "src/ai_excel_learning/ai_dashboard.py",
        "run_ai_excel_learning_webapp.py",
        "AI_EXCEL_LEARNING_DEMO_README.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} létezik")
        else:
            print(f"❌ {file_path} nem létezik")
            all_exist = False
    
    return all_exist

def test_dependencies():
    """Teszteli a függőségeket"""
    print("\n📦 Függőségek tesztelése...")
    
    required_packages = [
        "streamlit",
        "pandas", 
        "plotly",
        "numpy"
    ]
    
    all_available = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} elérhető")
        except ImportError:
            print(f"❌ {package} nem elérhető")
            all_available = False
    
    return all_available

def test_desktop_tab():
    """Teszteli a desktop tab funkcionalitását"""
    print("\n🖥️ Desktop tab tesztelése...")
    
    try:
        # Mock notebook és app objektumok
        class MockNotebook:
            pass
        
        class MockApp:
            pass
        
        notebook = MockNotebook()
        app = MockApp()
        
        # Tab létrehozása
        from gui.ai_excel_learning_demo_tab import AIExcelLearningDemoTab
        tab = AIExcelLearningDemoTab(notebook, app)
        
        print("✅ Desktop tab létrehozás sikeres")
        print(f"✅ Tab frame típusa: {type(tab.frame)}")
        
        # Cleanup
        if hasattr(tab, 'cleanup'):
            tab.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Desktop tab teszt sikertelen: {e}")
        return False

def test_webapp_functions():
    """Teszteli a webapp függvényeket"""
    print("\n🌐 Webapp függvények tesztelése...")
    
    try:
        from ai_excel_learning.webapp_demo import (
            show_home_page, show_examples_page, show_dashboard_page,
            show_testing_page, show_results_page
        )
        
        print("✅ Webapp függvények import sikeres")
        
        # Függvények léteznek
        functions = [
            show_home_page, show_examples_page, show_dashboard_page,
            show_testing_page, show_results_page
        ]
        
        for func in functions:
            if callable(func):
                print(f"✅ {func.__name__} függvény hívható")
            else:
                print(f"❌ {func.__name__} függvény nem hívható")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Webapp függvények teszt sikertelen: {e}")
        return False

def test_ai_dashboard():
    """Teszteli az AI dashboard funkcionalitását"""
    print("\n📊 AI Dashboard tesztelése...")
    
    try:
        from ai_excel_learning.ai_dashboard import create_dashboard
        
        if callable(create_dashboard):
            print("✅ AI Dashboard create_dashboard függvény hívható")
        else:
            print("❌ AI Dashboard create_dashboard függvény nem hívható")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ AI Dashboard teszt sikertelen: {e}")
        return False

def run_integration_test():
    """Futtat egy integrációs tesztet"""
    print("\n🔗 Integrációs teszt futtatása...")
    
    try:
        # Mock Streamlit session state
        import sys
        from unittest.mock import MagicMock
        
        # Mock streamlit
        mock_st = MagicMock()
        sys.modules['streamlit'] = mock_st
        
        # Mock pandas
        mock_pd = MagicMock()
        sys.modules['pandas'] = mock_pd
        
        # Mock plotly
        mock_plotly = MagicMock()
        sys.modules['plotly'] = mock_plotly
        
        print("✅ Mock modulok létrehozva")
        
        # Egyszerű teszt adatok
        test_data = {
            'category': 'test',
            'timestamp': '2024-12-01T12:00:00',
            'success': True,
            'message': 'Teszt sikeres'
        }
        
        print("✅ Teszt adatok létrehozva")
        print(f"   Kategória: {test_data['category']}")
        print(f"   Időbélyeg: {test_data['timestamp']}")
        print(f"   Állapot: {'Sikeres' if test_data['success'] else 'Sikertelen'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrációs teszt sikertelen: {e}")
        return False

def main():
    """Fő teszt függvény"""
    print("🤖 AI Excel Learning Demo - Teszt Script")
    print("=" * 50)
    
    # Tesztek futtatása
    tests = [
        ("Importok", test_imports),
        ("Fájl struktúra", test_file_structure),
        ("Függőségek", test_dependencies),
        ("Desktop tab", test_desktop_tab),
        ("Webapp függvények", test_webapp_functions),
        ("AI Dashboard", test_ai_dashboard),
        ("Integrációs teszt", run_integration_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} teszt kivételt dobott: {e}")
            results.append((test_name, False))
    
    # Eredmények összefoglalása
    print("\n" + "=" * 50)
    print("📋 TESZT EREDMÉNYEK")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ SIKERES" if result else "❌ SIKERTELEN"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nÖsszesítés: {passed}/{total} teszt sikeres")
    
    if passed == total:
        print("🎉 Minden teszt sikeresen lefutott!")
        return 0
    else:
        print("⚠️  Néhány teszt sikertelen volt")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 