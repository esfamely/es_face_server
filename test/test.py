from pathlib import Path

import cv2
from db.table_dict import TableDict

from main.setup import Setup

face_dir = Setup.s3_face_dir + "_tmp/"

lFile = list(Path(face_dir).glob("*.jpg"))
img_urls = [str(img_url).replace("\\", "/") for img_url in lFile]
sorted(img_urls, reverse=True)
img_urls.sort(reverse=True)
print(img_urls)

table_dict = TableDict()
hm = 3
sql = "select * from face_prediction where isdeleted = '0' order by id desc limit 0,%(hm)s"
p_list = table_dict.list(sql, {"hm": hm})
print(p_list)

img = cv2.imread("https://kutikomiya.jp/images/idol/a/asuka-kirara001.W120.jpg")
cv2.imshow("es", img)
cv2.waitKey(0)
