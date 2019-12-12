import requests
from rec_sys.common import *


def upload_item(uid, vid, tag):
    params = item_sample.copy()

    params["video_id"] = vid
    params["user_id"] = uid
    if tag is not None:
        params["tag"] = tag

    r = requests.post(BASE_URL + ITEM_UPLOAD_PATH, data=params)

    print(r.text)


def upload_opt(item_id, opt_type, w, location):
    params = item_opt_sample.copy()

    params["item_id"] = item_id
    params["opt_id"] = item_id + opt_type
    params["opt_type"] = opt_type
    params["weight"] = w
    params["location"] = location

    r = requests.post(BASE_URL + OPT_ITEM_UPLOAD_PATH, data=params)

    print(r.status_code, r.text)


if __name__ == '__main__':

    tags = ["bad luck", "afrobeat dance", "football", "Narrative", "spoof"]

    for i in range(150):
        tag = tags[i // 30]
        item_id = "item_id_" + str(i)
        user_id = "user_id_" + str(i)
        # upload_item(user_id, item_id, tag)
        upload_opt(item_id, "HOT", 0.9, i%10)