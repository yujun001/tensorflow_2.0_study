import time
import requests
from datetime import datetime

from rec_sys.const import *


def upload_items(data):
    url = "/".join([BASE_URL, UPLOAD_ITEMS_PATH])

    print("req url", url)

    r = requests.post(url=url, data=data, headers=FORM_URLENCODED)

    print(r.content)


AREAR_CC_MAP = {
    "AFRICA_EN": ["NG", "GH", "GM", "SL", "LR", "KE", "TZ", "SS", "UG", "ZA", "BW", "NA", "SZ", "LS", "MW", "ZM", "ZW"],
    "AFRICA_FR": ["CI", "SN", "ML", "BF", "GQ", "TG", "BJ", "NE", "BI", "CF", "GQ", "GA", "CG", "CD", "MG"],
    "AFRICA_EF": ["RW", "SC", "CM", "MU"],
    "AFRICA_FA": ["DJ", "TD", "KM"],
    "AFRICA_EA": ["ER"],
    "AFRICA_AB": ["MR", "SO"],
    "AFRICA_SM": ["GW", "CV", "ET", "ST", "AO", "MZ"],
    "NORTH_AFRICA_AB": ["EH", "EG", "SD", "LY", "TN", "DZ", "MA"],
}

if __name__ == '__main__':
    dtime = datetime.now()
    un_time = int(time.mktime(dtime.timetuple())) * 1000

    common_data = {
        "video_id": "video_id_test_01",
        "user_id": "user_id_test_00",
        "recommend": 1,
        "action_type": 1,
        "country_code": "CN",
        "exp": "true",
        "rec_status": 1,
        "created_time": un_time,
    }

    ts = int(time.time())

    for area, cc in AREAR_CC_MAP.items():

        for c in cc:
            ts += 1
            common_data["video_id"] = "_".join(["video_id", area, c, "%d" % (ts)])
            common_data["country_code"] = c

            print(common_data["video_id"])
            upload_items(common_data)
