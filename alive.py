import requests
import time
import logging
from datetime import datetime

# --- CONFIGURATION ---
# Replace this with your ACTUAL deployed Render URL
API_URL = "https://adityaagrawal-bitspilani.onrender.com/" 
# Render sleeps after 15 mins of inactivity, so we ping every 10 mins (600 seconds)
PING_INTERVAL_SECONDS = 600 

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def keep_alive():
    print(f"🚀 Starting Keep-Alive Monitor for: {API_URL}")
    print(f"⏱️  Ping interval: {PING_INTERVAL_SECONDS} seconds")
    print("------------------------------------------------")

    while True:
        try:
            # We assume your main.py has the @app.get("/") endpoint we wrote earlier
            response = requests.get(API_URL, timeout=10)
            
            if response.status_code == 200:
                logging.info(f"✅ Ping Success! Status: {response.status_code} | Latency: {response.elapsed.total_seconds()}s")
            else:
                logging.warning(f"⚠️ Ping Warning! Status: {response.status_code}")

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Ping Failed: {e}")

        # Sleep until next ping
        time.sleep(PING_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        keep_alive()
    except KeyboardInterrupt:
        print("\n🛑 Stopping Keep-Alive Monitor.")
