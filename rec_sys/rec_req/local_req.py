import requests
import json

from rec_sys.const import *
from rec_sys.rec_req.common import *


def rec_sys_req(data):
    url = "/".join([BASE_URL, REC_REQUEST])

    print("req url", url)

    r = requests.post(url=url, data=data, headers=FORM_URLENCODED)

    try:
        print(json.dumps(json.loads(r.content), indent=True))
    except:
        print(r.content)


if __name__ == '__main__':
    rec_sys_req(COMMON_REQ_DATA)
