# Heroku Procfile
# Specifies the commands Heroku should run

# Web dyno - runs the Flask application
web: gunicorn backend.app:app --workers 2 --threads 4 --timeout 120 --log-file -

# Optional: Worker dyno for background tasks (if using Celery)
# worker: celery -A backend.celery_app worker --loglevel=info

# Optional: Clock dyno for scheduled tasks
# clock: python backend/scheduler.py

