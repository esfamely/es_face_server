import os
from pathlib import Path

from db.table_dict import TableDict

from main.setup import Setup
from utils.utils_sys import id_generator, now_dt_str

pid = "1"
root_dir = Setup.s3_face_dir
dirs = os.listdir(root_dir)
dict_list = []
detail_dict_list = []
for dir in dirs:
    print(dir)
    cid = id_generator()
    uid = dir
    dict = {"id": cid, "pid": pid, "uid": uid, "ct": "1",
            "dt": now_dt_str(), "isdeleted": "0"}
    dict_list.append(dict)

    lFile = list(Path(os.path.join(root_dir, dir)).glob("*.jpg"))
    for file in lFile:
        file_name = str(file).split("\\")[-1].split(".")[0]
        print(file_name)
        detail_dict = {"id": file_name, "cid": cid, "pid": pid, "uid": uid,
                       "iid": file_name, "dt": now_dt_str()}
        detail_dict_list.append(detail_dict)

table_dict = TableDict()
if len(detail_dict_list) > 0:
    table_dict.batch_save("face_collection", dict_list)
    table_dict.batch_save("face_collection_detail", detail_dict_list)
