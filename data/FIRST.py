import csv

input_file = "elonmusk1.csv"     
output_file = "tweettimes.csv" 

# take orignal csv and make it more workable
# basically very complicated logic to simplify the date/time column and
# create the char_count and is_RT (is a retweet?) features
# Commas and new line characters in the tweet text from elonmusk.csv
# make this a complicated process but this script is designed to handle that
with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    writer.writerow(["date","time","char_count","is_RT"])
    char_count = 0
    is_rt = 0
    jan1 = False
    for row in reader:
        if len(row) >= 1 and (row[-1][-3:] == 'EST' or row[-1][-3:] == 'EDT'): 
            dt = row[-1].strip('"').split(", ")
            if dt[0] == "Jan 1": # this part is designed to extrapolate the date information 
                jan1 = True      # provided and add years through its implicit ordering
            if jan1:
                dt[0] += " 2025"
            else:
                dt[0] += " 2024"
            if len(row[0]) == 19 and row[0].isnumeric():
                text = "".join(row[1:-1])
                dt.append(len(text))
                if text[:4] == "RT @": # detect a retweet in text
                    is_rt = 1
            else:
                dt.append(char_count + len("".join(row[:-1])))
            dt.append(is_rt)
            char_count = 0
            is_rt = 0
            writer.writerow(dt)  
        elif len(row) > 0:
            if len(row[0]) == 19 and row[0].isnumeric() and len(row) > 1:
                text = "".join(row[1:])
                char_count += len(text)
                if text[:4] == "RT @":
                    is_rt = 1
            else:
                char_count += len("".join(row))
