#!/bin/bash
# Build and run the FreightRates backend + frontend containers on a shared Docker network
set -e

docker network create freight-net 2>/dev/null || true      # shared network (ignore if it exists)
docker rm -f freight-backend freight-frontend 2>/dev/null || true   # remove old containers

# Backend (Flask API) - reachable from the frontend as http://freight-backend:7860
docker build -t freight-backend ./backend_files
docker run -d --name freight-backend --network freight-net -p 7860:7860 freight-backend

# Frontend (Streamlit) - talks to the backend through the Docker network
docker build -t freight-frontend ./frontend_files
docker run -d --name freight-frontend --network freight-net -p 8501:8501 \
  -e BACKEND_URL=http://freight-backend:7860 freight-frontend

docker ps
echo "Backend  -> port 7860 | Frontend -> port 8501 (see the PORTS tab)"
