import psycopg2
import sys
import os
sys.path.append(os.getcwd())
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
conn.autocommit = True
cur = conn.cursor()
with open('database/schema.sql', 'r') as f:
    cur.execute(f.read())
print('Tables created successfully!')
conn.close()