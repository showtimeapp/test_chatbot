import pandas as pd
import sqlite3
import os

def csv_to_sqlite(csv_path='data/temple_data.csv', db_path='data/temple.db'):
    """Convert CSV file to SQLite database"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Clean column names (remove spaces, make lowercase)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    
    # Write dataframe to SQLite
    df.to_sql('temples', conn, if_exists='replace', index=False)
    
    # Create indexes for better query performance
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector ON temples(sector)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ward ON temples(ward)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deity ON temples(deity)")
    
    conn.commit()
    conn.close()
    
    print(f"Database created successfully at {db_path}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    
    return df

if __name__ == "__main__":
    csv_to_sqlite()

# import pandas as pd
# import sqlite3
# import os
# import sys

# def csv_to_sqlite(csv_path='data/temple_data.csv', db_path='data/temple.db'):
#     """Convert CSV file to SQLite database"""

#     # Create data directory if it doesn't exist
#     os.makedirs('data', exist_ok=True)

#     if not os.path.exists(csv_path):
#         print(f"❌ CSV file not found at {csv_path}. Skipping DB setup.")
#         return

#     try:
#         # Read CSV file
#         df = pd.read_csv(csv_path)

#         # Clean column names (remove spaces, make lowercase)
#         df.columns = [col.lower().replace(' ', '_') for col in df.columns]

#         # Connect to SQLite database
#         conn = sqlite3.connect(db_path)

#         # Write dataframe to SQLite (append instead of replace if you want to keep data)
#         df.to_sql('temples', conn, if_exists='replace', index=False)

#         # Create indexes for better query performance
#         cursor = conn.cursor()
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector ON temples(sector)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_ward ON temples(ward)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_deity ON temples(deity)")

#         conn.commit()
#         conn.close()

#         print(f"✅ Database created successfully at {db_path}")
#         print(f"   Total records: {len(df)}")
#         print(f"   Columns: {', '.join(df.columns)}")

#     except Exception as e:
#         print(f"❌ Error during DB setup: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     csv_to_sqlite()
#     sys.exit(0)
