#!/usr/bin/env python3
"""
Adatbázis inicializáló script a DataChaEnhanced projekthez
"""

import sqlite3
import os
from pathlib import Path

def init_background_tasks_db():
    """Inicializálja a háttér taskok adatbázisát"""
    
    db_path = Path("background_tasks.db")
    
    print(f"🗄️ Háttér taskok adatbázis inicializálása: {db_path}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Tasks tábla
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    file_path TEXT,
                    parameters TEXT,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    progress REAL DEFAULT 0.0,
                    result TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    schedule_time TEXT,
                    recurring INTEGER DEFAULT 0,
                    recurring_interval TEXT
                )
            """)
            
            # Notifications tábla
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    notification_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data TEXT,
                    read INTEGER DEFAULT 0
                )
            """)
            
            # Indexek
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_status ON tasks(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_priority ON tasks(priority DESC)
            """)
            
            conn.commit()
            
            print("✅ Tasks és notifications táblák létrehozva")
            
            # Teszt adatok hozzáadása
            from datetime import datetime
            
            # Teszt task
            test_task = {
                'task_id': 'test_task_001',
                'task_type': 'excel_analysis',
                'task_name': 'Teszt Excel Elemzés',
                'file_path': 'test_file.xlsx',
                'parameters': '{"analysis_type": "basic"}',
                'priority': 2,
                'status': 'completed',
                'created_at': datetime.now().isoformat(),
                'started_at': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat(),
                'progress': 100.0,
                'result': '{"success": true, "score": 95}',
                'error_message': None,
                'retry_count': 0,
                'max_retries': 3,
                'schedule_time': None,
                'recurring': 0,
                'recurring_interval': None
            }
            
            conn.execute("""
                INSERT OR REPLACE INTO tasks VALUES (
                    :task_id, :task_type, :task_name, :file_path, :parameters,
                    :priority, :status, :created_at, :started_at, :completed_at,
                    :progress, :result, :error_message, :retry_count, :max_retries,
                    :schedule_time, :recurring, :recurring_interval
                )
            """, test_task)
            
            # Teszt notification
            test_notification = {
                'task_id': 'test_task_001',
                'notification_type': 'task_completed',
                'message': 'Teszt task sikeresen befejezve',
                'timestamp': datetime.now().isoformat(),
                'data': '{"score": 95}',
                'read': 0
            }
            
            conn.execute("""
                INSERT INTO notifications (task_id, notification_type, message, timestamp, data, read)
                VALUES (:task_id, :notification_type, :message, :timestamp, :data, :read)
            """, test_notification)
            
            conn.commit()
            print("✅ Teszt adatok hozzáadva")
            
    except Exception as e:
        print(f"❌ Hiba a háttér taskok adatbázis inicializálásakor: {e}")
        raise

def init_ai_monitoring_db():
    """Inicializálja az AI monitoring adatbázist"""
    
    db_path = Path("ai_monitoring.db")
    
    print(f"🤖 AI monitoring adatbázis inicializálása: {db_path}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Metrics tábla
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT,
                    tags TEXT
                )
            """)
            
            # Alerts tábla
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolution_time TEXT,
                    context TEXT
                )
            """)
            
            # Recommendations tábla
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recommendation_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    implemented INTEGER DEFAULT 0,
                    context TEXT
                )
            """)
            
            conn.commit()
            print("✅ AI monitoring táblák létrehozva")
            
    except Exception as e:
        print(f"❌ Hiba az AI monitoring adatbázis inicializálásakor: {e}")
        raise

def init_research_db():
    """Inicializálja a kutatási adatbázist"""
    
    db_path = Path("research.db")
    
    print(f"🔬 Kutatási adatbázis inicializálása: {db_path}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Projects tábla
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # File analysis tábla
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    file_path TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    results TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)
            
            # Learning results tábla
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    model_name TEXT NOT NULL,
                    accuracy REAL,
                    training_time REAL,
                    parameters TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)
            
            conn.commit()
            print("✅ Kutatási táblák létrehozva")
            
    except Exception as e:
        print(f"❌ Hiba a kutatási adatbázis inicializálásakor: {e}")
        raise

def main():
    """Fő függvény"""
    
    print("🚀 DataChaEnhanced Adatbázis Inicializálás")
    print("=" * 50)
    
    try:
        # Háttér taskok adatbázis
        init_background_tasks_db()
        
        # AI monitoring adatbázis
        init_ai_monitoring_db()
        
        # Kutatási adatbázis
        init_research_db()
        
        print("\n🎉 Minden adatbázis sikeresen inicializálva!")
        print("\n📊 Létrehozott adatbázisok:")
        print("  • background_tasks.db - Háttér taskok és értesítések")
        print("  • ai_monitoring.db - AI monitoring és metrikák")
        print("  • research.db - Kutatási projektek és eredmények")
        
    except Exception as e:
        print(f"\n❌ Hiba történt: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 