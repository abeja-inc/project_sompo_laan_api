import http
import numpy as np
import json
import traceback
import time

def handler(request, ctx):

    try:
        json_data = open('./sample/weakness.json', 'r', encoding="utf-8")
        json_data = json.load(json_data)
        json_data['timestamp'] = int(time.time() * 1000)

        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': json_data
        }
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return {
            'status_code': http.HTTPStatus.INTERNAL_SERVER_ERROR,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': str(e)}
        }