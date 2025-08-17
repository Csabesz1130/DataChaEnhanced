#!/usr/bin/env python3
"""
Háttér Task Client - DataChaEnhanced

Ez a modul kliens interfészt biztosít a háttér task szolgáltatás használatához.
GUI és WebApp alkalmazások ezzel kommunikálnak a háttér szolgáltatással.

Funkciók:
- Task beküldés és követés
- Értesítések fogadása
- Service management
- Real-time status updates
"""

import os
import sys
import time
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .background_task_service import (
    BackgroundTask, TaskStatus, TaskPriority, 
    is_service_running, start_service_daemon, stop_service
)

logger = logging.getLogger(__name__)

class BackgroundTaskClient:
    """
    Háttér Task Client
    
    Ez az osztály egyszerű interfészt biztosít a háttér task szolgáltatás
    használatához GUI és WebApp alkalmazások számára.
    """
    
    def __init__(self, storage_dir: str = "background_tasks"):
        """
        Inicializálja a task clientet
        
        Args:
            storage_dir: A háttér szolgáltatás adatkönyvtára
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.db_path = self.storage_dir / "tasks.db"
        self.status_path = self.storage_dir / "service_status.json"
        
        # Notification callbacks
        self.notification_callbacks = []
        
        # Polling thread
        self.polling_thread = None
        self.polling_active = False
        self.last_notification_check = datetime.now()
        
        logger.info("BackgroundTaskClient inicializálva")
    
    def is_service_running(self) -> bool:
        """Ellenőrzi, hogy a háttér szolgáltatás fut-e"""
        return is_service_running()
    
    def start_service(self) -> bool:
        """Elindítja a háttér szolgáltatást ha nincs fut"""
        if self.is_service_running():
            logger.info("Háttér szolgáltatás már fut")
            return True
        
        try:
            success = start_service_daemon()
            if success:
                # Várunk egy kicsit, hogy a szolgáltatás elinduljon
                for _ in range(10):
                    time.sleep(0.5)
                    if self.is_service_running():
                        logger.info("Háttér szolgáltatás sikeresen elindítva")
                        return True
                
                logger.warning("Háttér szolgáltatás indítása után nem fut")
                return False
            else:
                logger.error("Háttér szolgáltatás indítása sikertelen")
                return False
                
        except Exception as e:
            logger.error(f"Háttér szolgáltatás indítási hiba: {e}")
            return False
    
    def stop_service(self) -> bool:
        """Leállítja a háttér szolgáltatást"""
        try:
            return stop_service()
        except Exception as e:
            logger.error(f"Háttér szolgáltatás leállítási hiba: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Lekéri a szolgáltatás állapotát"""
        try:
            if self.status_path.exists():
                with open(self.status_path, 'r') as f:
                    status_data = json.load(f)
                
                status_data['running'] = self.is_service_running()
                return status_data
            else:
                return {
                    'running': self.is_service_running(),
                    'status': 'unknown',
                    'active_tasks': 0,
                    'queue_size': 0,
                    'workers': 0
                }
                
        except Exception as e:
            logger.error(f"Service status lekérési hiba: {e}")
            return {
                'running': False,
                'status': 'error',
                'error': str(e)
            }
    
    def submit_excel_analysis_task(self, 
                                 file_path: str, 
                                 task_name: Optional[str] = None,
                                 priority: TaskPriority = TaskPriority.NORMAL,
                                 schedule_time: Optional[datetime] = None) -> str:
        """
        Excel elemzési task beküldése
        
        Args:
            file_path: Excel fájl elérési útja
            task_name: Task neve (opcionális)
            priority: Task prioritása
            schedule_time: Ütemezett futtatási idő (opcionális)
            
        Returns:
            Task ID
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel fájl nem található: {file_path}")
        
        if not task_name:
            task_name = f"Excel elemzés: {os.path.basename(file_path)}"
        
        task = BackgroundTask(
            task_id=f"excel_analysis_{int(time.time())}_{hash(file_path) % 10000}",
            task_type="excel_analysis",
            task_name=task_name,
            file_path=file_path,
            priority=priority,
            schedule_time=schedule_time
        )
        
        return self._submit_task(task)
    
    def submit_formula_learning_task(self,
                                   file_path: str,
                                   task_name: Optional[str] = None,
                                   priority: TaskPriority = TaskPriority.NORMAL,
                                   schedule_time: Optional[datetime] = None) -> str:
        """Képlet tanulási task beküldése"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel fájl nem található: {file_path}")
        
        if not task_name:
            task_name = f"Képlet tanulás: {os.path.basename(file_path)}"
        
        task = BackgroundTask(
            task_id=f"formula_learning_{int(time.time())}_{hash(file_path) % 10000}",
            task_type="formula_learning",
            task_name=task_name,
            file_path=file_path,
            priority=priority,
            schedule_time=schedule_time
        )
        
        return self._submit_task(task)
    
    def submit_chart_learning_task(self,
                                 file_path: str,
                                 task_name: Optional[str] = None,
                                 priority: TaskPriority = TaskPriority.NORMAL,
                                 schedule_time: Optional[datetime] = None) -> str:
        """Grafikon tanulási task beküldése"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel fájl nem található: {file_path}")
        
        if not task_name:
            task_name = f"Grafikon tanulás: {os.path.basename(file_path)}"
        
        task = BackgroundTask(
            task_id=f"chart_learning_{int(time.time())}_{hash(file_path) % 10000}",
            task_type="chart_learning",
            task_name=task_name,
            file_path=file_path,
            priority=priority,
            schedule_time=schedule_time
        )
        
        return self._submit_task(task)
    
    def submit_full_pipeline_task(self,
                                file_path: str,
                                task_name: Optional[str] = None,
                                priority: TaskPriority = TaskPriority.HIGH,
                                schedule_time: Optional[datetime] = None) -> str:
        """Teljes pipeline task beküldése"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel fájl nem található: {file_path}")
        
        if not task_name:
            task_name = f"Teljes pipeline: {os.path.basename(file_path)}"
        
        task = BackgroundTask(
            task_id=f"full_pipeline_{int(time.time())}_{hash(file_path) % 10000}",
            task_type="full_pipeline",
            task_name=task_name,
            file_path=file_path,
            priority=priority,
            schedule_time=schedule_time
        )
        
        return self._submit_task(task)
    
    def submit_batch_processing_task(self,
                                   file_list: List[str],
                                   task_name: Optional[str] = None,
                                   priority: TaskPriority = TaskPriority.NORMAL,
                                   schedule_time: Optional[datetime] = None) -> str:
        """Batch feldolgozási task beküldése"""
        # Ellenőrizzük, hogy az összes fájl létezik
        missing_files = [f for f in file_list if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Hiányzó fájlok: {missing_files}")
        
        if not task_name:
            task_name = f"Batch feldolgozás: {len(file_list)} fájl"
        
        task = BackgroundTask(
            task_id=f"batch_processing_{int(time.time())}_{len(file_list)}",
            task_type="batch_processing",
            task_name=task_name,
            parameters={"file_list": file_list},
            priority=priority,
            schedule_time=schedule_time
        )
        
        return self._submit_task(task)
    
    def submit_recurring_cleanup_task(self,
                                    interval: str = "daily",
                                    task_name: Optional[str] = None) -> str:
        """Ismétlődő cleanup task beküldése"""
        if not task_name:
            task_name = f"Rendszeres cleanup ({interval})"
        
        # Következő futtatási idő kiszámítása
        now = datetime.now()
        if interval == "daily":
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        elif interval == "weekly":
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
            days_ahead = 6 - now.weekday()  # Vasárnap
            if days_ahead <= 0:
                days_ahead += 7
            next_run += timedelta(days=days_ahead)
        elif interval == "monthly":
            next_run = now.replace(day=1, hour=2, minute=0, second=0, microsecond=0)
            if next_run.month == 12:
                next_run = next_run.replace(year=next_run.year + 1, month=1)
            else:
                next_run = next_run.replace(month=next_run.month + 1)
        else:
            raise ValueError(f"Ismeretlen interval: {interval}")
        
        task = BackgroundTask(
            task_id=f"cleanup_{interval}_{int(time.time())}",
            task_type="scheduled_cleanup",
            task_name=task_name,
            priority=TaskPriority.LOW,
            schedule_time=next_run,
            recurring=True,
            recurring_interval=interval
        )
        
        return self._submit_task(task)
    
    def _submit_task(self, task: BackgroundTask) -> str:
        """Task beküldése a szolgáltatásnak"""
        # Ellenőrizzük, hogy a szolgáltatás fut-e
        if not self.is_service_running():
            # Megpróbáljuk elindítani
            if not self.start_service():
                raise RuntimeError("Háttér szolgáltatás nem indítható el")
        
        # Task mentése adatbázisba
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO tasks (
                        task_id, task_type, task_name, file_path, parameters,
                        priority, status, created_at, started_at, completed_at,
                        progress, result, error_message, retry_count, max_retries,
                        schedule_time, recurring, recurring_interval
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.task_id, task.task_type, task.task_name, task.file_path,
                    json.dumps(task.parameters) if task.parameters else None,
                    task.priority.value, task.status.value,
                    task.created_at.isoformat(),
                    task.started_at.isoformat() if task.started_at else None,
                    task.completed_at.isoformat() if task.completed_at else None,
                    task.progress,
                    json.dumps(task.result) if task.result else None,
                    task.error_message, task.retry_count, task.max_retries,
                    task.schedule_time.isoformat() if task.schedule_time else None,
                    1 if task.recurring else 0, task.recurring_interval
                ))
                conn.commit()
            
            logger.info(f"Task beküldve: {task.task_id}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Task beküldési hiba: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Task állapot lekérdezése"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        "task_id": row['task_id'],
                        "task_type": row['task_type'],
                        "task_name": row['task_name'],
                        "file_path": row['file_path'],
                        "status": row['status'],
                        "progress": row['progress'],
                        "created_at": row['created_at'],
                        "started_at": row['started_at'],
                        "completed_at": row['completed_at'],
                        "error_message": row['error_message'],
                        "priority": row['priority']
                    }
                
        except Exception as e:
            logger.error(f"Task állapot lekérdezési hiba: {e}")
        
        return None
    
    def get_all_tasks(self, 
                     status_filter: Optional[str] = None,
                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Összes task lekérdezése"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                
                if status_filter:
                    query = "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC"
                    params = (status_filter,)
                else:
                    query = "SELECT * FROM tasks ORDER BY created_at DESC"
                    params = ()
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query, params)
                
                tasks = []
                for row in cursor.fetchall():
                    tasks.append({
                        "task_id": row['task_id'],
                        "task_type": row['task_type'],
                        "task_name": row['task_name'],
                        "file_path": row['file_path'],
                        "status": row['status'],
                        "progress": row['progress'],
                        "created_at": row['created_at'],
                        "started_at": row['started_at'],
                        "completed_at": row['completed_at'],
                        "priority": row['priority']
                    })
                
                return tasks
                
        except Exception as e:
            logger.error(f"Task lista lekérdezési hiba: {e}")
            return []
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Aktív taskok lekérdezése"""
        return self.get_all_tasks(status_filter="running")
    
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Pending taskok lekérdezése"""
        return self.get_all_tasks(status_filter="pending")
    
    def get_completed_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Befejezett taskok lekérdezése"""
        return self.get_all_tasks(status_filter="completed", limit=limit)
    
    def get_failed_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Sikertelen taskok lekérdezése"""
        return self.get_all_tasks(status_filter="failed", limit=limit)
    
    def cancel_task(self, task_id: str) -> bool:
        """Task megszakítása"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    UPDATE tasks SET status = 'cancelled' 
                    WHERE task_id = ? AND status IN ('pending', 'scheduled')
                """, (task_id,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Task megszakítva: {task_id}")
                    return True
                else:
                    logger.warning(f"Task nem szakítható meg: {task_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Task megszakítási hiba: {e}")
            return False
    
    def get_notifications(self, unread_only: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        """Értesítések lekérdezése"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                
                if unread_only:
                    query = "SELECT * FROM notifications WHERE read = 0 ORDER BY timestamp DESC"
                else:
                    query = "SELECT * FROM notifications ORDER BY timestamp DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query)
                
                notifications = []
                for row in cursor.fetchall():
                    notifications.append({
                        "id": row['id'],
                        "task_id": row['task_id'],
                        "notification_type": row['notification_type'],
                        "message": row['message'],
                        "timestamp": row['timestamp'],
                        "data": json.loads(row['data']) if row['data'] else {},
                        "read": bool(row['read'])
                    })
                
                return notifications
                
        except Exception as e:
            logger.error(f"Értesítések lekérdezési hiba: {e}")
            return []
    
    def mark_notification_read(self, notification_id: int) -> bool:
        """Értesítés olvasottnak jelölése"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    UPDATE notifications SET read = 1 WHERE id = ?
                """, (notification_id,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    return True
                
        except Exception as e:
            logger.error(f"Értesítés jelölési hiba: {e}")
        
        return False
    
    def mark_all_notifications_read(self) -> int:
        """Összes értesítés olvasottnak jelölése"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("UPDATE notifications SET read = 1 WHERE read = 0")
                conn.commit()
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Értesítések jelölési hiba: {e}")
            return 0
    
    def add_notification_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Értesítési callback hozzáadása"""
        self.notification_callbacks.append(callback)
        
        # Ha ez az első callback, indítsuk el a polling-ot
        if len(self.notification_callbacks) == 1 and not self.polling_active:
            self.start_notification_polling()
    
    def remove_notification_callback(self, callback: Callable):
        """Értesítési callback eltávolítása"""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
        
        # Ha nincs több callback, állítsuk le a polling-ot
        if len(self.notification_callbacks) == 0 and self.polling_active:
            self.stop_notification_polling()
    
    def start_notification_polling(self, interval: int = 2):
        """Értesítés polling indítása"""
        if self.polling_active:
            return
        
        self.polling_active = True
        self.polling_thread = threading.Thread(
            target=self._notification_polling_loop,
            args=(interval,),
            daemon=True
        )
        self.polling_thread.start()
        
        logger.info("Értesítés polling elindítva")
    
    def stop_notification_polling(self):
        """Értesítés polling leállítása"""
        if not self.polling_active:
            return
        
        self.polling_active = False
        
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)
        
        logger.info("Értesítés polling leállítva")
    
    def _notification_polling_loop(self, interval: int):
        """Értesítés polling loop"""
        while self.polling_active:
            try:
                # Új értesítések keresése
                new_notifications = self._get_new_notifications()
                
                # Callback-ek hívása
                for notification in new_notifications:
                    for callback in self.notification_callbacks:
                        try:
                            callback(notification)
                        except Exception as e:
                            logger.error(f"Notification callback hiba: {e}")
                
                # Következő ellenőrzés időpontjának frissítése
                if new_notifications:
                    self.last_notification_check = datetime.now()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Notification polling hiba: {e}")
                time.sleep(5)
    
    def _get_new_notifications(self) -> List[Dict[str, Any]]:
        """Új értesítések lekérdezése az utolsó ellenőrzés óta"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM notifications 
                    WHERE timestamp > ? 
                    ORDER BY timestamp ASC
                """, (self.last_notification_check.isoformat(),))
                
                notifications = []
                for row in cursor.fetchall():
                    notifications.append({
                        "id": row['id'],
                        "task_id": row['task_id'],
                        "notification_type": row['notification_type'],
                        "message": row['message'],
                        "timestamp": row['timestamp'],
                        "data": json.loads(row['data']) if row['data'] else {},
                        "read": bool(row['read'])
                    })
                
                return notifications
                
        except Exception as e:
            logger.error(f"Új értesítések lekérdezési hiba: {e}")
            return []
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Task statisztikák lekérdezése"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT 
                        status,
                        COUNT(*) as count,
                        AVG(CASE WHEN status = 'completed' THEN 
                            (julianday(completed_at) - julianday(started_at)) * 86400 
                            ELSE NULL END) as avg_duration
                    FROM tasks 
                    GROUP BY status
                """)
                
                status_counts = {}
                avg_durations = {}
                
                for row in cursor.fetchall():
                    status_counts[row[0]] = row[1]
                    if row[2] is not None:
                        avg_durations[row[0]] = row[2]
                
                # Összes task számtása
                total_tasks = sum(status_counts.values())
                
                # Sikerességi arány
                completed = status_counts.get('completed', 0)
                failed = status_counts.get('failed', 0)
                success_rate = (completed / (completed + failed) * 100) if (completed + failed) > 0 else 0
                
                return {
                    "total_tasks": total_tasks,
                    "status_counts": status_counts,
                    "success_rate": success_rate,
                    "avg_duration_seconds": avg_durations.get('completed', 0),
                    "service_running": self.is_service_running()
                }
                
        except Exception as e:
            logger.error(f"Statisztikák lekérdezési hiba: {e}")
            return {
                "total_tasks": 0,
                "status_counts": {},
                "success_rate": 0,
                "avg_duration_seconds": 0,
                "service_running": self.is_service_running()
            }
    
    def cleanup_old_tasks(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Régi taskok törlése"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # Régi befejezett taskok törlése
                cursor1 = conn.execute("""
                    DELETE FROM tasks 
                    WHERE status IN ('completed', 'failed', 'cancelled') 
                    AND created_at < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_tasks = cursor1.rowcount
                
                # Régi értesítések törlése
                cursor2 = conn.execute("""
                    DELETE FROM notifications 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_notifications = cursor2.rowcount
                
                conn.commit()
                
                logger.info(f"Cleanup befejezve: {deleted_tasks} task, {deleted_notifications} értesítés törölve")
                
                return {
                    "deleted_tasks": deleted_tasks,
                    "deleted_notifications": deleted_notifications
                }
                
        except Exception as e:
            logger.error(f"Cleanup hiba: {e}")
            return {"deleted_tasks": 0, "deleted_notifications": 0}
    
    def export_task_results(self, output_path: str, task_ids: Optional[List[str]] = None) -> bool:
        """Task eredmények exportálása"""
        try:
            if task_ids:
                placeholders = ','.join(['?' for _ in task_ids])
                query = f"SELECT * FROM tasks WHERE task_id IN ({placeholders})"
                params = task_ids
            else:
                query = "SELECT * FROM tasks WHERE status = 'completed'"
                params = ()
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "tasks": []
                }
                
                for row in cursor.fetchall():
                    task_data = dict(row)
                    # JSON mezők parsáolása
                    if task_data['parameters']:
                        task_data['parameters'] = json.loads(task_data['parameters'])
                    if task_data['result']:
                        task_data['result'] = json.loads(task_data['result'])
                    
                    export_data["tasks"].append(task_data)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Task eredmények exportálva: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Export hiba: {e}")
            return False
    
    def __del__(self):
        """Destruktor - polling leállítása"""
        if self.polling_active:
            self.stop_notification_polling()


# Convenience functions
def create_background_task_client(storage_dir: str = "background_tasks") -> BackgroundTaskClient:
    """Egyszerű factory function a client létrehozásához"""
    return BackgroundTaskClient(storage_dir)

def submit_excel_file_for_analysis(file_path: str, 
                                 priority: TaskPriority = TaskPriority.NORMAL,
                                 schedule_time: Optional[datetime] = None) -> str:
    """
    Gyors Excel fájl elemzés beküldése
    
    Ez a function automatikusan létrehoz egy clientet, beküld egy taskot
    és visszaadja a task ID-t.
    """
    client = create_background_task_client()
    return client.submit_excel_analysis_task(file_path, priority=priority, schedule_time=schedule_time)

def submit_excel_files_batch(file_paths: List[str],
                            priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """Gyors batch feldolgozás beküldése"""
    client = create_background_task_client()
    return client.submit_batch_processing_task(file_paths, priority=priority)

def get_task_status_simple(task_id: str) -> Optional[Dict[str, Any]]:
    """Egyszerű task státusz lekérdezés"""
    client = create_background_task_client()
    return client.get_task_status(task_id)

def get_recent_completed_tasks(limit: int = 10) -> List[Dict[str, Any]]:
    """Legutóbbi befejezett taskok lekérdezése"""
    client = create_background_task_client()
    return client.get_completed_tasks(limit=limit)

if __name__ == "__main__":
    # Test script
    client = create_background_task_client()
    
    print("Háttér Task Client Teszt")
    print("=" * 40)
    
    # Service status
    status = client.get_service_status()
    print(f"Service fut: {status.get('running', False)}")
    
    # Task statistics
    stats = client.get_task_statistics()
    print(f"Összes task: {stats['total_tasks']}")
    print(f"Sikerességi arány: {stats['success_rate']:.1f}%")
    
    # Recent tasks
    recent_tasks = client.get_all_tasks(limit=5)
    print(f"\nLegutóbbi {len(recent_tasks)} task:")
    for task in recent_tasks:
        print(f"  {task['task_id']}: {task['status']} - {task['task_name']}")
    
    # Notifications
    notifications = client.get_notifications(unread_only=True, limit=5)
    print(f"\nOlvasatlan értesítések: {len(notifications)}")
    for notif in notifications:
        print(f"  {notif['notification_type']}: {notif['message']}")