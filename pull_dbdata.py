import pandas as pd
import psycopg2

# ---- DB CONFIG ----

DB_HOST = "159.198.44.98"
DB_NAME = "neuroheart"
DB_USER = "neuroheart_user"
DB_PASSWORD = "YOUR_PASSWORD"
DB_PORT = 5432

# ---- QUERY ----

query = """
SELECT
id,
user_id,
sample_type,
start_time,
end_time,
value,
unit,
payload
FROM health_samples
ORDER BY start_time DESC
LIMIT 50000
"""

# ---- CONNECT ----

conn = psycopg2.connect(
host=DB_HOST,
dbname=DB_NAME,
user=DB_USER,
password=DB_PASSWORD,
port=DB_PORT
)

print("Connected to DB")

df = pd.read_sql(query, conn)

# ---- SAVE CSV ----

output_file = "health_samples_latest.csv"
df.to_csv(output_file, index=False)

print(f"Saved {len(df)} rows to {output_file}")

conn.close()

# python pull_dbdata.py>s