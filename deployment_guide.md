# Deploying Firm Bankruptcy Prediction App

Since your application is built with **Streamlit** (Python), it cannot be hosted on GitHub Pages (which only supports static HTML/CSS/JS).

The best and easiest way to host this project for free is **Streamlit Cloud**.

## Prerequisites (Completed)
- [x] Project pushed to GitHub
- [x] `requirements.txt` present
- [x] File paths updated for cloud compatibility (I just did this for you!)

## Steps to Deploy on Streamlit Cloud

1.  **Sign Up / Login**:
    *   Go to [share.streamlit.io](https://share.streamlit.io/)
    *   Click "Continue with GitHub"

2.  **New App**:
    *   Click the "New app" button (top right).
    *   Select "Use existing repo".

3.  **Configure**:
    *   **Repository**: Select `gattiuday/firm_bankruptcy_prediction`
    *   **Branch**: `main`
    *   **Main file path**: `app.py` (It should auto-detect this)

4.  **Deploy**:
    *   Click "Deploy!"
    *   Wait a few minutes for it to install dependencies and start.

## Troubleshooting

*   **"FileNotFoundError"**: If you see this, it means the paths are still wrong. I updated `app.py` to use relative paths like `data/processed/...` which should work perfectly.
*   **"Module not found"**: Ensure all libraries are in `requirements.txt`. I checked and it looks correct (`xgboost`, `streamlit`, `scikit-learn`, etc. are there).

## Alternative: Render (Docker)

If you prefer to use the Docker container:
1.  Sign up at [render.com](https://render.com).
2.  New "Web Service".
3.  Connect GitHub repo.
4.  Select "Docker" as the runtime.
5.  Deploy. (Note: The free tier on Render spins down after inactivity).
