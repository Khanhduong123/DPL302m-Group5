import sqlite3

def price(name):
  conn = sqlite3.connect('instance/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%{name}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchone()
  result = rows
  conn.close()
  return result

def chatbot(name):
  result = price(name)
  answer = f"sản phẩm {result[0]}có giá {result[1]} đồng"
  return answer
 