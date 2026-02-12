import requests
import time
from inference.run_inference import infer
import json

PATH_UNITY = "unity_data.json"


class CricketCommentarySystem:
    def __init__(self, json_url):
        self.json_url = json_url
        self.processed_ball_ids = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_data(self):
        try:
            r = requests.get(self.json_url, headers=self.headers, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"Connection Error: {e}")
        return None

    def process_stream(self):
        data = self.fetch_data()
        if not data: 
            return []

        raw_commentary = data.get("commentary", []) # send to nlp
        
        # Sort Oldest -> Newest based on OID (Over ID) or ID
        # Using OID as primary sort key usually handles chronological order best
        sorted_balls = sorted(raw_commentary, key=lambda x: float(x.get('OID', 0)))

        if not sorted_balls:
            return []

        # Determine Live Position (Last ball in the sorted list)
        latest_ball = sorted_balls[-1]
        try:
            current_live_val = float(latest_ball.get("Over", "0.0"))
        except ValueError:
            return []

        # Sliding Window Threshold (Live - 1.0)
        threshold_val = current_live_val - 1.0
        
        # Output list containing pairs of (Text, Meta)
        new_updates = []

        for ball in sorted_balls:
            ball_id = ball.get("Id")
            
            if ball_id in self.processed_ball_ids:
                continue

            try:
                ball_val = float(ball.get("Over", "0.0"))
            except ValueError:
                continue

            if ball_val <= threshold_val:
                comm_text = ball.get("Commentary") or ball.get("Default_Commentary", "")
                
                # Extract Metadata specific to THIS ball
                meta = {
                    "runs": ball.get("Runs", "0"),
                    "over": ball.get("Over", ""),
                    "speed": ball.get("Ball_Speed", "N/A"),
                    "batsman_hand": ball.get("Batsman_Style", "N/A") 
                }
                
                # Add to processed set
                self.processed_ball_ids.add(ball_id)
                
                # Append the PAIR to the output
                new_updates.append({
                    "text": comm_text,
                    "meta": meta
                })

        return new_updates

# ==========================================
# EXECUTION
# ==========================================

URL = "https://www.hindustantimes.com/static-content/10s/commentary_268068_2.json"  

system = CricketCommentarySystem(URL)

print("Starting System (Corrected Data Sync)...")

while True:
    updates = system.process_stream()
    
    if updates:
        print(f"\n--- {len(updates)} NEW BALLS RELEASED ---")
        for item in updates:
            # Now we print the metadata SPECIFIC to this commentary line
            
            data = item['meta'] 
            temp = infer(item['text'], item['meta']['over'], 'models')
            data |= temp['predictions']  # Merge predictions into metadata

            with open(PATH_UNITY, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)

        
            print(f"Commentary: {item['text']}")
            print(f"Metadata:   {item['meta']}")
            print("-" * 30)
            
    time.sleep(5)