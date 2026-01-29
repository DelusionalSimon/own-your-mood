import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Config
DAYS_BACK = 5
REC_DIR = Path("recordings")
REC_DIR.mkdir(exist_ok=True)

print(f"🌱 Seeding {DAYS_BACK} days of history...")

for i in range(DAYS_BACK):
    # Calculate date (Today, Yesterday, etc.)
    date_obj = datetime.now() - timedelta(days=i)
    
    # Create a dummy timestamp
    ts = date_obj.timestamp()
    
    # 1. Create a dummy .wav file
    filename = f"test_recording_day_{i}.wav"
    filepath = REC_DIR / filename
    
    with open(filepath, 'wb') as f:
        f.write(b'\x00' * 1024) # 1KB of silence
        
    # 2. CRITICAL: Backdate the file modification time
    os.utime(filepath, (ts, ts))
    
    

    print(f"  Created entry for: {date_obj.strftime('%Y-%m-%d')}")

print("\nDone! Restart the app to see your 🔥 5 Day Streak.")