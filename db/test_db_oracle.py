import pandas as pd
import cx_Oracle
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

host = '192.168.1.131'
port = '1521'
sid = 'orcl'
dsn = cx_Oracle.makedsn(host, port, sid)
conn = cx_Oracle.connect('nhsl', 'nhsl', dsn)

sql = 'select * from BASEPLAM_USERS'
results = pd.read_sql(sql, conn)
print('{}, {}'.format(results.shape[0], results.at[0, 'USERNAME']))

'''cursor = conn.cursor()
sql = "insert into BASEPLAM_USERS(ID, LOGINID, PASSWD, USERNAME) values ('%s', '%s', '%s', '%s')"
cursor.execute(sql % ('123456', 'sam', '123', '呵呵'))
conn.commit()

sql = "update BASEPLAM_USERS set PASSWD = 'ko' where ID = :id"
cursor.execute(sql, {'id': '123456'})
conn.commit()

cursor.close()'''
conn.close()
