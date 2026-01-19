# UniView API Gateway

Simple JWT authentication gateway for the UniView system.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment file
cp .env.example .env

# 3. Run the gateway
python main.py
```

Gateway runs on: http://localhost:8080

## Test It

```bash
# Login
curl -X POST http://localhost:8080/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}'

# Use the token from response
curl -X GET http://localhost:8080/api/protected \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Environment Variables

Edit `.env`:
- `JWT_SECRET_KEY` - Change this to a secure random string
- `BACKEND_URL` - Your backend service URL (default: http://localhost:8000)
- `FRONTEND_URL` - Your frontend URL (default: http://localhost:3000)

## API Endpoints

**Public:**
- `GET /` - Service info
- `GET /health` - Health check
- `POST /api/login` - Login (get JWT token)

**Protected (requires JWT):**
- `GET /api/protected` - Example protected route
- `GET /api/backend-health` - Check backend status

## Documentation

Interactive API docs: http://localhost:8080/docs

## Demo Credentials

- Username: `demo`
- Password: `demo123`

**Remove these before production!**
