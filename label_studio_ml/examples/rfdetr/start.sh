#!/bin/bash
exec gunicorn --bind :${PORT:-9091} --workers ${WORKERS:-1} --threads ${THREADS:-4} --timeout 0 _wsgi:app
