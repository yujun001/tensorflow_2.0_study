import requests

from rec_sys.common import *




def upload_action(uid, unique_id, vid, alg, rec_id):
    params = action_sample.copy()

    if uid is not None:
        params["user_id"] = uid

    if unique_id is not None:
        params["unique_id"] = unique_id

    params["alg"] = alg
    params["rec_id"] = rec_id
    params["object_id"] = vid

    r = requests.post(BASE_URL + ACTION_UPLOAD, data=params)

    print(r.status_code, r.text)


if __name__ == '__main__':

    for i in range(20):
        uid = None
        unique_id = "763b9290-a156-4e70-8cb1-d16a41f079eb_002"
        vid = "item_id_%2d" % (i)
        alg = "9002"
        rec_id = "1a924aef7b1a46f6b291d3cb5d03209d_0"
        upload_action(uid, unique_id, vid, alg, rec_id)
