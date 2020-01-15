import pandas as pd
import pymysql
from main.setup import Setup


class TableDict:
    """
    dict对象的数据库存取
    """

    def __init__(self):
        self.conn = None

    def open(self):
        self.conn = pymysql.connect(Setup.s0_db_ip,
                                    Setup.s0_db_login, Setup.s0_db_pw, Setup.s0_db_name)

    def close(self):
        self.conn.close()

    def list(self, sql, value_dict=None):
        """
        sql查询
        :return: result_dict_list
        """
        self.open()

        if value_dict is None:
            results = pd.read_sql(sql, self.conn)
        else:
            results = pd.read_sql(sql, self.conn, params=value_dict)

        self.close()

        result_dict_list = []
        size_int = results.shape[0]
        for i in range(size_int):
            result_dict = {}
            for column in results:
                result_dict[column] = results.at[i, column]
            result_dict_list.append(result_dict)

        return result_dict_list

    def get(self, table, id):
        """
        读取一个dict对象
        """
        result_dict_list = self.list("select * from {} where id = %(id)s".format(table), {"id": id})
        if len(result_dict_list) == 0:
            return None
        else:
            return result_dict_list[0]

    def save(self, table, dict):
        """
        新增一个dict对象
        """
        dict_list = [dict]
        self.batch_save(table, dict_list)

    def update(self, table, dict):
        """
        更新一个dict对象
        """
        dict_list = [dict]
        self.batch_update(table, dict_list)

    def batch_save(self, table, dict_list):
        """
        新增多个dict对象
        """
        self.open()
        cursor = self.conn.cursor()

        for i, dict in enumerate(dict_list):
            sql = "insert into {} (".format(table)
            value_list = []
            for i, item in enumerate(dict):
                sql += ("{}" if i == 0 else ", {}").format(item)
            sql += ") values ("
            for i, item in enumerate(dict):
                sql += "%s" if i == 0 else ", %s"
                value_list.append(dict[item])
            sql += ")"

            cursor.execute(sql, value_list)

        self.conn.commit()

        cursor.close()
        self.close()

    def batch_update(self, table, dict_list):
        """
        更新多个dict对象
        """
        self.open()
        cursor = self.conn.cursor()

        for i, dict in enumerate(dict_list):
            sql = "update {} set ".format(table)
            value_list = []
            i = 0
            for item in dict:
                if item == "id":
                    continue
                sql += "{}{} = %s".format(("" if i == 0 else ", "), item)
                value_list.append(dict[item])
                i += 1
            sql += " where id = %s"
            value_list.append(dict["id"])

            cursor.execute(sql, value_list)

        self.conn.commit()

        cursor.close()
        self.close()

    def exec(self, sql):
        """
        执行sql
        """
        sql_list = [sql]
        self.batch_exec(sql_list)

    def batch_exec(self, sql_list):
        """
        执行多个sql
        """
        self.open()

        cursor = self.conn.cursor()
        for sql in sql_list:
            cursor.execute(sql)
        self.conn.commit()

        cursor.close()
        self.close()
