import os
import http
import traceback
from abeja.training import APIClient

# 入力チャンネルの取得
JOB_DEFINITION_ID = os.getenv('JOB_DEFINITION_ID', '')
JOB_DEFINITION_NAME = os.getenv('JOB_DEFINITION_NAME', '')
JOB_DEFINITION_VERSION = os.getenv('JOB_DEFINITION_VERSION', 0)

def handler(request, ctx):

    try:

        api_client = APIClient()
        response = api_client.create_training_job(JOB_DEFINITION_ID,JOB_DEFINITION_NAME,JOB_DEFINITION_VERSION,None,None)

        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': 'OK'}
        }
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return {
            'status_code': http.HTTPStatus.INTERNAL_SERVER_ERROR,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': str(e)}
        }