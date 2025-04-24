import os
import streamlit.web.cli as stcli
import sys

def app():
    # Run Streamlit with frontend/app.py, listening on Render's PORT
    sys.argv = [
        "streamlit",
        "run",
        "frontend/app.py",
        "--server.port",
        os.getenv("PORT", "8501"),
        "--server.address",
        "0.0.0.0"
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    app()
