import sqlite3

def price(name):
  conn = sqlite3.connect('instance/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT phone_money FROM thongtindienthoai WHERE phone_names LIKE '%{name}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchone()
  result = rows[0]
  conn.close()
  return result