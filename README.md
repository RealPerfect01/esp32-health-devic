# MedMon Backend (FastAPI)

## Run locally
    docker build -t medmon-api .
    docker run -p 8000:8000 --env-file .env.example medmon-api
    # docs: http://localhost:8000/docs

## Deploy on Render
1. Push this folder to a GitHub repo.
2. Render dashboard > New > Blueprint > pick the repo (uses render.yaml).
3. Enter ADMIN_PASSWORD when prompted. Deploy.
4. Copy DEVICE_API_KEY from the service's Environment tab (used by the ESP32).

## Frontend usage
    curl -X POST $URL/api/auth/login -H "Content-Type: application/json" \
         -d '{"username":"admin","password":"..."}'
    curl $URL/api/readings?limit=50 -H "Authorization: Bearer <token>"
    curl $URL/api/readings/latest -H "Authorization: Bearer <token>"
    curl $URL/api/devices -H "Authorization: Bearer <token>"

Query params for /api/readings: device_name, since, until (ISO 8601), only_valid, limit (<=1000), offset.
