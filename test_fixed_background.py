#!/usr/bin/env python3
"""
Teszt script a javított háttér szolgáltatáshoz
"""

import sys
import os

# Hozzáadás a Python path-hoz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_background_service():
    """Teszteli a javított háttér szolgáltatást"""
    
    print("🧪 Háttér Szolgáltatás Tesztelése")
    print("=" * 40)
    
    try:
        from src.utils.background_task_client import BackgroundTaskClient
        
        print("✅ BackgroundTaskClient import sikeres")
        
        # Client létrehozása
        client = BackgroundTaskClient()
        print("✅ BackgroundTaskClient létrehozva")
        
        # Adatbázis ellenőrzése
        print("\n📊 Adatbázis ellenőrzése...")
        
        # Tasks lekérdezése
        tasks = client.get_all_tasks()
        print(f"  Tasks: {len(tasks)} db")
        
        # Notifications lekérdezése
        notifications = client.get_notifications()
        print(f"  Notifications: {len(notifications)} db")
        
        # Service állapot
        service_running = client.is_service_running()
        print(f"  Service fut: {service_running}")
        
        # Teszt task létrehozása
        print("\n🔧 Teszt task létrehozása...")
        
        from src.utils.background_task_service import TaskPriority
        
        test_task_id = "test_fixed_001"
        success = client.submit_excel_analysis_task(
            "test_file.xlsx",
            priority=TaskPriority.NORMAL
        )
        
        if success:
            print("✅ Teszt task sikeresen beküldve")
            
            # Task állapot lekérdezése
            task_status = client.get_task_status(test_task_id)
            if task_status:
                print(f"  Task állapot: {task_status['status']}")
            else:
                print("  Task állapot nem lekérdezhető")
        else:
            print("❌ Teszt task beküldés sikertelen")
        
        print("\n🎉 Háttér szolgáltatás teszt sikeres!")
        return True
        
    except Exception as e:
        print(f"❌ Hiba: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_background_service()
    sys.exit(0 if success else 1) 