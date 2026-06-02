# Deployment Guide — Railway

## Prerequisites
- Railway account: https://railway.app
- Railway CLI: `npm install -g @railway/cli`
- GitHub repo connected to Railway

## Environment Variables (set in Railway dashboard)

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |
| `PORT` | Set automatically by Railway |

> ⚠️ Never commit `.env` to git. Always set secrets via the Railway dashboard.

## Deploy via Railway CLI

```bash
# 1. Login
railway login

# 2. Link to your project (first time)
railway link

# 3. Set your Groq key
railway variables set GROQ_API_KEY=gsk_your_key_here

# 4. Deploy
railway up

# 5. Open in browser
railway open
```

## Deploy via GitHub (recommended)

1. Go to https://railway.app/new
2. Choose **Deploy from GitHub repo**
3. Select `ofrywaits/ai-course-analytics`
4. Set `GROQ_API_KEY` in the **Variables** tab
5. Railway auto-deploys on every push to `main`

## Files used by Railway

| File | Purpose |
|------|---------|
| `Procfile` | Tells Railway how to start the app |
| `railway.json` | Build and deploy configuration |
| `.streamlit/config.toml` | Streamlit server settings for production |
| `requirements.txt` | Python dependencies |

## Verify deployment

After deploy:
1. Open the Railway URL (e.g. `https://ai-course-analytics.up.railway.app`)
2. You should see the Streamlit dashboard
3. Click **Run Full Analysis** to trigger the pipeline
4. Check Railway logs for any errors: `railway logs`

## Monitoring on Railway

- **Logs**: `railway logs --tail` streams live logs
- **Metrics**: CPU/memory graphs in the Railway dashboard
- **Health**: The platform's own health banner shows file status
