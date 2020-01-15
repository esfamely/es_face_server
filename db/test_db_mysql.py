import pandas as pd
import pymysql
from utils.utils_sys import id_generator, now_dt_str

conn = pymysql.connect("127.0.0.1", "es", "123456", "es_face")
#print(conn)

sql = 'select * from face_collection'
results = pd.read_sql(sql, conn)
print(results)
#print('{}, {}'.format(results.shape[0], results.at[0, 'width']))

cursor = conn.cursor()
sql = "insert into face_collection(id, uid, pid, width, height, dt, isdeleted)"
sql += " values ('%s', '%s', '%s', %d, %d, '%s', '%s')"
cursor.execute(sql % (id_generator(), "20", "1", 160, 160, now_dt_str(), "0"))
conn.commit()

sql = "update face_collection set pid = '15' where width = %(width)s"
cursor.execute(sql, {"width": 160})
conn.commit()

cursor.close()
conn.close()
