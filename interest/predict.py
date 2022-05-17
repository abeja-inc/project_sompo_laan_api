import http
import numpy as np
import json
import traceback
import time
from io import BytesIO

def handler(request, ctx):

    try:
        ##リクエストの取得
        r = request.read()
        b = BytesIO(r)
        j = json.loads(b.read().decode('utf-8'))

        ret = {'result':[]}

        for id in j['learner_id']:
            json_data = open('./sample/interest.json', 'r', encoding="utf-8")
            json_data = json.load(json_data)
            json_data['timestamp'] = int(time.time() * 1000)
            json_data['personal']['id'] = id
            ret['result'].append(json_data)

        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': ret
        }
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return {
            'status_code': http.HTTPStatus.INTERNAL_SERVER_ERROR,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': str(e)}
        }