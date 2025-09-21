import pandas as pd
import sqlite3
import os

def setup_databases():
    """Convert CSV files to SQLite database with multiple tables"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect('data/data.db')
    cursor = conn.cursor()
    
    # Setup temples table if CSV exists
    temple_csv = 'data/temple_data.csv'
    if os.path.exists(temple_csv):
        df_temples = pd.read_csv(temple_csv)
        df_temples.columns = [col.lower().replace(' ', '_') for col in df_temples.columns]
        df_temples.to_sql('temples', conn, if_exists='replace', index=False)
        
        # Create indexes for temples
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temple_sector ON temples(sector)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temple_ward ON temples(ward)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temple_deity ON temples(deity)")
        
        print(f"✓ Temples table created with {len(df_temples)} records")
        print(f"  Columns: {', '.join(df_temples.columns)}")
    else:
        print(f"⚠ Temple CSV not found at {temple_csv}")
    
    # Setup schools table if CSV exists
    school_csv = 'data/school_data.csv'
    if os.path.exists(school_csv):
        df_schools = pd.read_csv(school_csv)
        df_schools.columns = [col.lower().replace(' ', '_') for col in df_schools.columns]
        
        # Convert camelCase to snake_case
        new_columns = []
        for col in df_schools.columns:
            # Convert camelCase to snake_case
            import re
            col = re.sub('([A-Z]+)', r'_\1', col).lower()
            col = col.lstrip('_')
            new_columns.append(col)
        df_schools.columns = new_columns
        
        df_schools.to_sql('schools', conn, if_exists='replace', index=False)
        
        # Create indexes for schools
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_school_sector ON schools(sector)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_school_ward ON schools(ward)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_school_board ON schools(board)")
        
        print(f"✓ Schools table created with {len(df_schools)} records")
        print(f"  Columns: {', '.join(df_schools.columns)}")
    else:
        print(f"⚠ School CSV not found at {school_csv}")
    
    conn.commit()
    conn.close()
    
    print("\n✓ Database setup complete at data/data.db")

if __name__ == "__main__":
    setup_databases()