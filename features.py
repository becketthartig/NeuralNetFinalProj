''' 
time of day, time left in week, total tweets in day/week, 
moving averages, n-characters, frequency
'''
import csv
from datetime import datetime

def time_to_secs(ts):
    ts = ts.rsplit(" ", 1)[0]
    dt = datetime.strptime(ts, "%b %d, %I:%M:%S %p")
    return dt.hour*3600 + dt.minute*60 + dt.second, dt.date()

output = [["time_of_day", "day_total"]]

with open("elonmusk.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    last_date = None
    day_total = 0
    
    for row in reader:
        if len(row) != 3:
            continue #Skip messed up rows
        ts_raw = row["created_at"]
        if not ts_raw:
            continue 
        
        try:
            time_secs, tweet_date = time_to_secs(ts_raw)
        except ValueError:
            continue

        if tweet_date != last_date:
            day_total = 1
            last_date = tweet_date
        else:
            day_total += 1
        
        output.append([time_secs, day_total])

with open("features.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(output)
