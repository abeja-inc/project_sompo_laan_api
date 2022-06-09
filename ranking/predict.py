import os
import http
import json
import traceback
from io import BytesIO
from abeja.datalake import Client as DatalakeClient

# 入力チャンネルの取得
INPUT_CHANNEL_ID = os.getenv('INPUT_CHANNEL_ID', '')

def handler(request, ctx):

    try:
        #学習データの取得
        data = learnig_data(INPUT_CHANNEL_ID)
        
        ##リクエストの取得
        r = request.read()
        b = BytesIO(r)
        j = json.loads(b.read().decode('utf-8'))
        id_list = j['learner_id']

        #リクエストされたIDのデータのみ取得
        result = []
        for r in data['result']:
            if(r['personal']['id'] in id_list):
                result.append(r)

        data['result'] = result

        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': data
        }
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return {
            'status_code': http.HTTPStatus.INTERNAL_SERVER_ERROR,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': str(e)}
        }

#対象学習データの取得
def learnig_data(channel_id):
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(channel_id)
    files = channel.list_files(sort='-uploaded_at')

    for i, f in enumerate(files):
        file_item = channel.get_file(f.file_id)
        file_name = file_item.metadata['filename']
        if('ranking' in file_name):
            print(file_name)
            data = json.load(BytesIO(file_item.get_content()))
            break
    return data