from db.table_dict import TableDict

from utils.utils_sys import id_generator


class CameraService:
    """
    终端机
    """

    def get_by_cn(self, cn):
        """
        取得某名称的终端机记录
        """
        table_dict = TableDict()
        sql = "select * from face_collection_camera where cn = %(cn)s"
        list = table_dict.list(sql, {"cn": cn})
        return None if len(list) <= 0 else list[0]

    def get_by_ip(self, ip):
        """
        取得某ip的终端机记录
        """
        table_dict = TableDict()
        sql = "select * from face_collection_camera where ip = %(ip)s"
        list = table_dict.list(sql, {"ip": ip})
        return None if len(list) <= 0 else list[0]

    def get_by_sn(self, sn):
        """
        取得某序号的终端机记录
        """
        table_dict = TableDict()
        sql = "select * from face_collection_camera where sn = %(sn)s"
        list = table_dict.list(sql, {"sn": sn})
        return None if len(list) <= 0 else list[0]

    def list_camera(self):
        """
        列出全部终端机
        """
        table_dict = TableDict()
        sql = "select * from face_collection_camera where isdeleted = '0' order by sn"
        c_list = table_dict.list(sql)
        for c in c_list:
            c["cno"] = str(c["cno"])
        return c_list

    def get_camera(self, id):
        """
        取得终端机信息
        """
        table_dict = TableDict()
        camera = table_dict.get("face_collection_camera", id)
        if camera is None:
            return {}
        camera["cno"] = str(camera["cno"])
        return camera

    def save_camera(self, id, cn, ip, sn, cno, tips):
        """
        新增或修改终端机
        """
        camera_cn = self.get_by_cn(cn)
        camera_ip = self.get_by_ip(ip)
        camera_sn = self.get_by_sn(sn)

        table_dict = TableDict()
        table = "face_collection_camera"
        id = id if id is not None else id_generator()
        camera = table_dict.get(table, id)
        if camera is None:
            # 确保名称、ip与序号的唯一
            if camera_cn is not None:
                return "", "已经有名称为“{}”的终端机，不能重复！".format(cn)
            if camera_ip is not None:
                return "", "已经有ip地址为“{}”的终端机，不能重复！".format(ip)
            if camera_sn is not None:
                return "", "已经有序号为“{}”的终端机，不能重复！".format(sn)

            # 新增
            camera = {"id": id, "cn": cn, "ip": ip, "sn": sn, "cno": cno, "tips": tips,
                      "isdeleted": "0"}
            table_dict.save(table, camera)
        else:
            # 确保名称、ip与序号的唯一
            if camera_cn is not None and camera_cn["id"] != id:
                return "", "已经有名称为“{}”的终端机，不能重复！".format(cn)
            if camera_ip is not None and camera_ip["id"] != id:
                return "", "已经有ip地址为“{}”的终端机，不能重复！".format(ip)
            if camera_sn is not None and camera_sn["id"] != id:
                return "", "已经有序号为“{}”的终端机，不能重复！".format(sn)

            # 修改
            camera["cn"] = cn
            camera["ip"] = ip
            camera["sn"] = sn
            camera["cno"] = cno
            camera["tips"] = tips
            table_dict.update(table, camera)

        return id, ""

    def del_camera(self, id_list):
        """
        删除终端机
        """
        if id_list is None or id_list.strip() == '':
            return
        table_dict = TableDict()
        sql_list = []
        id_lists = id_list.split(",")
        for id in id_lists:
            sql = "update face_collection_camera set isdeleted = '1' where id = '{}'".format(id)
            sql_list.append(sql)
        table_dict.batch_exec(sql_list)
