
DEV_BASE = "http://localhost:8080"

ACTION_UPLOAD = "/upload/action.s"

REC_PATH = "/rec/rec.s"

ITEM_UPLOAD_PATH = "/upload/item.s"

OPT_ITEM_UPLOAD_PATH = "/upload/rec/operation.s"

BASE_URL = DEV_BASE

item_sample = {
    "video_id": "video_id_x",
    "user_id": "user_id_x",
    "title": "title_x",
    "tag": "bad luck",
    "category": "category_x",
    "video_url": "video_url_x",
    "picture_url": "picture_url_x",
    "created_time": 1562169600000,
    "duration": 15000,
    "recommend": 1,
    "activity_id": "activity_id_x",
    "activity_title": "activity_title_x",
    "music_id": "music_id_x",
    "views": 100,
    "likes": 20,
    "shares": 1,
    "comment_count": 10,
    "action_type": 1,
    "play_complete": 100,
    "exp": "true"
}

item_opt_sample = {
    "opt_id": "opt_id_x",
    "item_id": "item_id_x",
    "opt_type": "FIX",
    "weight": 0.99,
    "operator": "operator_x",
    "start_time": 1562169600000,
    "end_time": "1582169600000",
    "status": 1,
    "exp": "true",
    "location": 10
}

action_sample = {
    "alg": "alg_x",
    "rec_id": "rec_id_x",
    "scene": "f",
    "object_id": "object_id_x",
    "rating": 2,
    "play_num": 2,
    "cost": 15000,
    "progress": 0.6,
    "exp": "true",
    "version": "version_x"
}

rec_sample = {
    "scene": "f",
    "num": 5,
    "os_version": "os_version_x",
    "ua": "ua_x",
    "country": "CN",
    "net_type": "WIFI",
    "exp": "true",
    "version": "1.0",
}
