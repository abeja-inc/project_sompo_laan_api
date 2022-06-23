import os
import http
import traceback
import requests
import json
from abeja.training import APIClient

# 入力チャンネルの取得
JOB_DEFINITION_ID = os.getenv('JOB_DEFINITION_ID', '')
JOB_DEFINITION_NAME = os.getenv('JOB_DEFINITION_NAME', '')
JOB_DEFINITION_VERSION = os.getenv('JOB_DEFINITION_VERSION', 0)
WEB_HOOK_URL = os.getenv('WEB_HOOK_URL', '')
OUTPUT_RECORDE_SIZE = os.getenv('OUTPUT_RECORDE_SIZE', 10)

def handler(request, ctx):

    try:

        api_client = APIClient()
        user_parameters = {'OUTPUT_RECORDE_SIZE': OUTPUT_RECORDE_SIZE}
        response = api_client.create_training_job(JOB_DEFINITION_ID,JOB_DEFINITION_NAME,JOB_DEFINITION_VERSION,user_parameters,None)

        post_slack('trigger_start:'+ JOB_DEFINITION_NAME + '\ntraining_job_id:' + response['training_job_id'])

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

#処理結果のSlack通知
def post_slack(message):
    requests.post(WEB_HOOK_URL, data = json.dumps({
        'text': message,  #通知内容
    }))