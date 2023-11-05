import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('instance/giasanpham.db')
cursor = conn.cursor()

# Execute the SQL statement to drop the table
cursor.execute('DROP TABLE IF EXISTS thongtindienthoai')

# Commit the changes and close the connection
conn.commit()
conn.close()