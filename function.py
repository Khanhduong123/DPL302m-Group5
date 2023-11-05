import sqlite3

def price(name):
  conn = sqlite3.connect('/content/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT phone_money FROM thongtindienthoai WHERE phone_names LIKE '%{name}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchone()
  result = rows[0]
  conn.close()
  return result

def resolution(name):
  conn = sqlite3.connect('/content/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT phone_screens FROM thongtindienthoai WHERE phone_names LIKE '%{name}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchone()
  result = rows[0]
  conn.close()
  return result

def memory(name):
  conn = sqlite3.connect('/content/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT phone_memories FROM thongtindienthoai WHERE phone_names LIKE '%{name}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchone()
  result = rows[0]
  conn.close()
  return result

def exist(name):
  conn = sqlite3.connect('/content/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT phone_memories FROM thongtindienthoai WHERE phone_names LIKE '%{name}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchall()
  if len(rows) == 0:
    result = "No"
  else :
    result = "Yes"
  return result

if exist(phone_name) == "Yes":
  if tag == "price_tag":
    result = price(phone_name)
    print(f"sản phẩm {phone_name} có giá {result}")
  if tag == "resolution_tag":
    result = resolution(phone_name)
    print(f"màn hình của sản phẩm {phone_name} có kích thước {result}")
else:
  print(f"sản phẩm {phone_name} không tồn tại hoặc hiện đang không kinh doanh bên cửa hàng chúng tôi")