# Render Deployment Guide

## Deploy to Render (Free Tier)

### Option 1: Single Service (Backend serves Frontend) - RECOMMENDED

This is simpler and uses only one free service.

#### Steps:

1. **Push your code to GitHub** (already done ✅)

2. **Sign up for Render**: Go to https://render.com and sign up with GitHub

3. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `ayanalamMOON/multilingual-search-engine`
   - Configure:
     - **Name**: `multilingual-search-engine`
     - **Region**: Oregon (US West) or closest to you
     - **Branch**: `main`
     - **Runtime**: `Python 3`
     - **Build Command**:
       ```bash
       pip install -r requirements.txt && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" && cd webui && npm install && npm run build
       ```
     - **Start Command**:
       ```bash
       uvicorn api:app --host 0.0.0.0 --port $PORT
       ```
     - **Plan**: `Free`

4. **Add Environment Variables**:
   - `PYTHON_VERSION` = `3.11.0`
   - `VECTOR_BACKEND` = `faiss`
   - `FAISS_INDEX_PATH` = `/opt/render/project/src/artifacts/faiss_index`
   - `INCLUDE_ENGLISH` = `1`
   - `TRANSFORMERS_CACHE` = `/opt/render/project/src/.cache`
   - `HF_HOME` = `/opt/render/project/src/.cache`

5. **Add Persistent Disk** (for FAISS index):
   - In Advanced settings
   - Click "Add Disk"
   - **Name**: `faiss-storage`
   - **Mount Path**: `/opt/render/project/src/artifacts`
   - **Size**: `1 GB`

6. **Deploy**: Click "Create Web Service"

7. **Wait**: First deployment takes ~10-15 minutes (downloading model)

8. **Access**: Your app will be at `https://multilingual-search-engine.onrender.com`

---

### Option 2: Separate Services (Backend + Frontend)

If you prefer separate services:

#### Backend Service:
1. Create Web Service with settings from Option 1 (without frontend build)
2. Build Command: `./build.sh`
3. Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

#### Frontend Service:
1. Create Static Site
2. Build Command: `cd webui && npm install && npm run build`
3. Publish Directory: `webui/dist`
4. Add rewrite rules:
   - `/api/*` → `https://your-backend-url.onrender.com/api/*`

---

## Important Notes

### Free Tier Limitations:
- ⚠️ **Spins down after 15 min inactivity** - First request after sleep takes ~30s
- 💾 **512 MB RAM** - Might be tight for the ML model; upgrade to Starter ($7/mo) for 512MB+ if needed
- 📦 **Free disk**: 1GB included
- 🕐 **750 hours/month** - Enough for hobby projects

### Performance Tips:
1. **Keep service alive**: Use a service like [UptimeRobot](https://uptimerobot.com/) to ping every 14 minutes
2. **First load is slow**: Model loads on first request (~10-20s)
3. **Persistent storage**: FAISS index persists across deployments
4. **Logs**: Check Render logs if deployment fails

### Troubleshooting:

**Build fails (out of memory)**:
- Reduce `DATASET_LIMIT` in environment variables to load fewer documents
- Remove English corpus: set `INCLUDE_ENGLISH=0`

**App crashes**:
- Check logs in Render dashboard
- Upgrade to paid plan for more RAM (512MB → 2GB)

**Slow responses**:
- Normal on free tier after wake-up
- Consider upgrading or using smaller model

---

## Alternative: Use render.yaml

If you prefer infrastructure-as-code, Render can auto-detect `render.yaml`:

1. Push code with `render.yaml` to GitHub
2. Go to Render Dashboard
3. Click "New +" → "Blueprint"
4. Select your repository
5. Render auto-configures from `render.yaml`

---

## Post-Deployment

### Update Backend URL in Frontend (if using separate services):
In `webui/src/App.jsx`, update:
```javascript
const API_BASE = import.meta.env.VITE_API_URL || 'https://your-backend.onrender.com'
```

### Set Environment Variable for Frontend Build:
Add to Render frontend service:
```
VITE_API_URL=https://your-backend.onrender.com
```

### Monitor:
- Health: `https://your-app.onrender.com/api/health`
- API Docs: `https://your-app.onrender.com/docs`
- Frontend: `https://your-app.onrender.com/`

---

**Estimated Deployment Time**: 10-15 minutes for first build

**Cost**: $0/month (free tier) or $7/month (Starter tier for better performance)
