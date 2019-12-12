import json

import requests

from rec_sys.common import *
from rec_sys.action import *


def recall(uid, unique_id):
    params = rec_sample.copy()

    # "unique_id": "unique_id_x",
    # "user_id": "user_id_x",

    if uid is not None:
        params["user_id"] = uid

    if unique_id is not None:
        params["unique_id"] = unique_id

    r = requests.post(BASE_URL + REC_PATH, data=params)

    # print("--------------------------->")
    # print("http status code ", r.status_code)

    return r.json()


def test_rec_num():
    fp = open("/home/daniel/backup/user_canrec.json", "r")

    js: dict = json.load(fp)

    for gaid in js.keys():
        res = recall(None, gaid)
        dst_alg = js[gaid]
        # try:
        data = res["data"]
        is_match = False
        for d in data:
            alg = d["alg"]
            if alg == dst_alg:
                is_match = True
                break

        if not is_match:
            print("gaid is wrong: " + gaid + " " + dst_alg)
        #
        # except e:
        #     print(e)
        #     print("gaid is except: " + gaid + " " + dst_alg)
        #     continue


if __name__ == '__main__':
    unique_id = "9b475e1c-f2be-4ca6-af58-4bbff8eaebff"
    vid = "vid_029"

    # test_rec_num()

    res = recall(None, unique_id)
    print(json.dumps(res, indent=True))


    # for i in range(5):
    #     upload_action(None, unique_id, vid, "9002", "d6f0363c3f6c478b973c7f3454e5a9f3_0")
