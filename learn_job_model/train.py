from abeja.datalake import Client as DatalakeClient
import pandas as pd
import numpy as np
import io
import time
import json
import os
import traceback
import requests

# 出力ディレクトリの作成
# ABEJA_TRAINING_RESULT_DIRという環境変数に出力先ディレクトリが設定される
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', '.')
os.makedirs(ABEJA_TRAINING_RESULT_DIR, exist_ok=True)

#環境変数の取り込み
INPUT_CHANNEL_ID = os.getenv('INPUT_CHANNEL_ID', 0)
OUTPUT_CHANNEL_ID = os.getenv('OUTPUT_CHANNEL_ID', 0)
INTEREST_COEFFICIENT = os.getenv('INTEREST_COEFFICIENT', 1)
WEAKNESS_COEFFICIENT = os.getenv('WEAKNESS_COEFFICIENT', 1)
WEB_HOOK_URL = os.getenv('WEB_HOOK_URL', '')

#処理IDの取得
TIMESTAMP = int(time.time() * 1000)


def handler(context):
    print('Start train handler.')
    print(ABEJA_TRAINING_RESULT_DIR)
    try:

        #マスタファイルの読み込み
        df_user, df_article, df_keyword, df_role, df_skill, df_level = input_datalake_master()

        #interestデータ作成
        user_vec = usertovec(df_user, df_keyword, df_role, df_skill, df_level, INTEREST_COEFFICIENT, WEAKNESS_COEFFICIENT)
        article_vec = articletovec(df_article, df_keyword, df_role, df_skill, df_level, 1, 1)
        #内積処理
        interest = calculation_dot(user_vec, article_vec)

        #weaknessデータ作成
        user_vec = usertovec(df_user, df_keyword, df_role, df_skill, df_level, WEAKNESS_COEFFICIENT, INTEREST_COEFFICIENT)
        #内積処理
        weakness = calculation_dot(user_vec, article_vec)

        #rankingデータ生成
        ranking = make_ranking(df_user, df_article)

        #ファイル出力
        output_file({'interest':interest, 'weakness':weakness, 'ranking':ranking})
        post_slack('process_id:' + str(TIMESTAMP) + '\nresult:Success')
        print('End train handler.')
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        post_slack('process_id:' + str(TIMESTAMP) + '\nresult:Failure' + '\nerror:' + str(e))
        raise e

#データレイクからマスタファイルの読み込み
def input_datalake_master():
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(INPUT_CHANNEL_ID)
    files = channel.list_files()
    for i, f in enumerate(files):
        file_item = channel.get_file(f.file_id)
        file_name = file_item.metadata['filename']
        if file_name == 'UserData.csv':
            _user = pd.read_csv(io.BytesIO(file_item.get_content()), encoding='utf-8', header=0)
        elif  file_name == 'ArticleData.csv':
            _article = pd.read_csv(io.BytesIO(file_item.get_content()), encoding='utf-8', header=0)
        elif  file_name == 'KeywordMaster.csv':
            _keyword = pd.read_csv(io.BytesIO(file_item.get_content()), encoding='utf-8', header=0)
        elif  file_name == 'RoleMaster.csv':
            _role = pd.read_csv(io.BytesIO(file_item.get_content()), encoding='utf-8', header=0)
        elif  file_name == 'SkillMaster.csv':
            _skill = pd.read_csv(io.BytesIO(file_item.get_content()), encoding='utf-8', header=0)
        elif  file_name == 'LevelMaster.csv':
            _level = pd.read_csv(io.BytesIO(file_item.get_content()), encoding='utf-8', header=0)
    return _user, _article, _keyword, _role, _skill, _level

#記事詳細データの取り出し
def get_article(id, content):
    return content[content["id"] == id]

#ファイル出力
def output_file(obj):
    for k, v in obj.items():
        metadata = { 'filename': str(TIMESTAMP) + '_' + k + '.json' }
        d  = {'timestamp':TIMESTAMP, 'recommend_type':k, 'result':v}
        output_datalake(d, metadata)
        output_dir(k, v)

#ディレクトリへの出力
def output_dir(recommend_type, obj):
    with open(os.path.join(ABEJA_TRAINING_RESULT_DIR, recommend_type + '.json'), 'w') as f:
        json.dump(obj, f, default=expireEncoda, ensure_ascii=False)

#指定のデータレイクチャンネルへのファイル出力
def output_datalake(object, metadata):
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(OUTPUT_CHANNEL_ID)
    channel.upload(json.dumps(object, default=expireEncoda, ensure_ascii=False).encode("utf-8"), metadata=metadata, content_type='application/json')

#オブジェクトのエンコード
def expireEncoda(object):
    if isinstance(object, np.integer):
        return int(object)

#ユーザーのベクトル化
def usertovec(user, keyword, role, skill, level, interest_coefficient, weakness_coefficient):
    user_vec = {'result':[]}
    for data in user.itertuples():
        #興味関心
        user_interest = [data.interest1, data.interest2, data.interest3]
        #ユーザーに紐づくスキル
        user_skill1 = [data.skill1, data.skill2, data.skill3, data.skill4]
        #役割
        user_role = [data.role]
        #ロールに紐づくスキル
        skills = role[role['title'] == user_role[0]]
        user_skill2 = [skills['skill1'].values[0], skills['skill2'].values[0], skills['skill3'].values[0], skills['skill4'].values[0]]
        #レベル
        user_level = [data.level]

        #ベクトル作成
        result = np.where(keyword["title"].isin(user_interest), 1*interest_coefficient, 0)
        result = np.append(result,np.where(skill["title"].isin(user_skill1), 1*weakness_coefficient, 0))
        result = np.append(result,np.where(role["title"].isin(user_role), 1, 0))
        result = np.append(result,np.where(skill["title"].isin(user_skill2), 1*weakness_coefficient, 0))
        result = np.append(result,np.where(level["title"].isin(user_level), 1, 0))
        dict_obj = {'id':data.id, 'vec':result.tolist()}
        user_vec['result'].append(dict_obj)
    return user_vec

#コンテンツのベクトル化
def articletovec(article, keyword, role, skill, level, interest_coefficient, weakness_coefficient):
    article_vec = {'result':[]}
    for data in article.itertuples():
        #興味関心
        art_interest = [data.interest1, data.interest2, data.interest3]
        #スキル
        art_skill = [data.skill1, data.skill2, data.skill3]
        #スキルを含む役割
        r = role[(role['skill1'].isin(art_skill))|(role['skill2'].isin(art_skill))|(role['skill3'].isin(art_skill))|(role['skill4'].isin(art_skill))]
        art_role = []
        for v in r['title'].values:
            art_role.append(v)
        #レベル
        art_level = [data.level]

        #ベクトル作成
        result = np.where(keyword["title"].isin(art_interest), 1, 0)
        result = np.append(result,np.where(skill["title"].isin(art_skill), 1, 0))
        result = np.append(result,np.where(role["title"].isin(art_role), 1, 0))
        result = np.append(result,np.where(skill["title"].isin(art_skill), 1, 0))
        result = np.append(result,np.where(level["title"].isin(art_level), 1, 0))
        dict_obj = {'id':data.id, 'vec':result.tolist()}
        article_vec['result'].append(dict_obj)
    return article_vec

#内積の算出
def calculation_dot(user, article):
    data = []
    for u in user['result']:
        ret =[]
        for a in article['result']:
            v1 = np.array(u['vec'], dtype=float)
            v2 = np.array(a['vec'], dtype=float)
            ret.append({'id':a['id'], 'score':np.dot(v1, v2)})
        #内積結果の降順
        ret = sorted(ret, key=lambda x: (-x['score'],x['id']))
        data.append({'personal': { 'id': u['id'] },'articles':ret})  

    #優先順位付け
    for d in data:
        priority = 1
        for a in d['articles']:
            a['priority'] = priority
            priority = priority + 1
    return data

#ランキングデータの作成
def make_ranking(user, article):
    data=[]
    smp_article = article.sample(frac=1)
    for u in user.itertuples():
        ret =[]
        priority = 1
        for a in smp_article.itertuples():
            ret.append({'id':a.id, 'score':0, 'priority':priority})
            priority = priority + 1
        data.append({'personal': { 'id': u.id },'articles':ret})
    return data

#処理結果のSlack通知
def post_slack(message):
    requests.post(WEB_HOOK_URL, data = json.dumps({
        'text': message,  #通知内容
    }))


if __name__ == '__main__':
    handler(None)