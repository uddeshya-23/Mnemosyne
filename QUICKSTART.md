# Project Mnemosyne - Quick Start Guide

## ✅ Prerequisites Check
- [x] Docker installed (version 27.3.1)
- [x] Docker Compose installed (v2.29.7)
- [ ] Groq API Key obtained from https://console.groq.com/

## 🚀 Setup Steps

### Step 1: Set Your Groq API Key

You need to set your Groq API key as an environment variable. Choose your platform:

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="gsk_your_actual_key_here"
```

**Linux/Mac:**
```bash
export GROQ_API_KEY=gsk_your_actual_key_here
```

**Alternatively**, you can edit the `.env` file in the project root:
```bash
# Edit .env file
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 2: Build and Start the Services

```bash
docker-compose up --build
```

This will:
1. Build the Python brain container (~2-3 minutes)
2. Build the Rust proxy container (~5-7 minutes)
3. Start both services with health checks
4. Create a network bridge for inter-service communication

**Expected Output:**
```
mnemosyne-brain | INFO:     Application startup complete.
mnemosyne-proxy | Mnemosyne Proxy listening on 0.0.0.0:8080
```

### Step 3: Verify Services Are Running

Open a new terminal and check health:

```bash
# Check proxy health
curl http://localhost:8080/health

# Check brain health
curl http://localhost:5000/health
```

Expected responses:
```json
{"status":"healthy","service":"mnemosyne-proxy"}
{"status":"healthy","active_sessions":0}
```

### Step 4: Run the Test Suite

```bash
python tests/attack_sim.py
```

This will run three tests:
1. ✅ Normal traffic flow to Groq
2. ✅ Jailbreak attack detection
3. ✅ Latency benchmark

### Step 5: Try It Yourself!

Send a chat request:

```bash
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-8192",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ]
  }'
```

## 🐛 Troubleshooting

### Issue: "GROQ_API_KEY must be set"
**Solution:** Make sure you've set the environment variable before running `docker-compose up`.

### Issue: Build fails on Rust compilation
**Solution:** The Rust build can take 5-10 minutes. Be patient. If it fails, try:
```bash
docker-compose build --no-cache proxy
```

### Issue: Brain service won't start
**Solution:** Check logs:
```bash
docker logs mnemosyne-brain
```

### Issue: Port already in use
**Solution:** Stop any services using ports 5000 or 8080:
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8080 | xargs kill -9
```

## 🛑 Stopping the Services

```bash
# Stop and remove containers
docker-compose down

# Stop, remove containers, and clean up volumes
docker-compose down -v
```

## 📊 Monitoring

View logs in real-time:

```bash
# All services
docker-compose logs -f

# Just the proxy
docker-compose logs -f mnemosyne-proxy

# Just the brain
docker-compose logs -f mnemosyne-brain
```

## 🎯 Next Steps

1. **Experiment with different prompts** - Try benign and malicious inputs
2. **Adjust the threshold** - Edit `brain/titans.py` to tune sensitivity
3. **Monitor surprise scores** - Check brain logs to see detection patterns
4. **Try different Groq models** - Use mixtral-8x7b-32768 or other models

## 📚 Additional Resources

- Full documentation: See `README.md`
- API reference: See `README.md` API Usage section
- Architecture details: See `implementation_plan.md`

---

**Ready to secure your LLM traffic! 🛡️**
