import sqlite3

DB_PATH = "face_log.db"

# Connect to database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Fetch and display all logs
cursor.execute("SELECT * FROM face_log")
rows = cursor.fetchall()

print("ID | Name    | Count | Timestamp")
print("---------------------------------")
for row in rows:
    print(f"{row[0]} | {row[1]} | {row[2]} ")

conn.close()
