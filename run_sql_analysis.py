import sqlite3

# 1️⃣ Connect to your database
conn = sqlite3.connect("database/healthcare.db")
cursor = conn.cursor()

# 2️⃣ Read the SQL file
with open("scripts/sql_analysis.sql", "r", encoding="utf-8") as f:
    sql_queries = f.read().split(";")  # split queries by semicolon

# 3️⃣ Execute each query and print results
for i, query in enumerate(sql_queries):
    query = query.strip()
    if query:  # skip empty lines
        print(f"\n--- Query {i+1} ---")
        print(query)
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            for row in results:
                print(row)
        except Exception as e:
            print("Error:", e)

# 4️⃣ Close the database connection
conn.close()
