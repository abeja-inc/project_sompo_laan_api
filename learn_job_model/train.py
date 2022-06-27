from abeja.datalake import Client as DatalakeClient
import pandas as pd
import numpy as np
import io
import time
import json
import os
import traceback
import requests
import dateutil.parser
import datetime
import zipfile
import collections

# 出力ディレクトリの作成
# ABEJA_TRAINING_RESULT_DIRという環境変数に出力先ディレクトリが設定される
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', '.')
os.makedirs(ABEJA_TRAINING_RESULT_DIR, exist_ok=True)

#処理当日日付を取得(JST)
t_delta = datetime.timedelta(hours=9)
jst = datetime.timezone(t_delta, 'JST') 
TODAY = datetime.datetime.now(jst) 

#環境変数の取り込み
INPUT_CHANNEL_ID = os.getenv('INPUT_CHANNEL_ID', 0)
OUTPUT_CHANNEL_ID = os.getenv('OUTPUT_CHANNEL_ID', 0)
INTEREST_COEFFICIENT = os.getenv('INTEREST_COEFFICIENT', 1)
WEAKNESS_COEFFICIENT = os.getenv('WEAKNESS_COEFFICIENT', 1)
WEB_HOOK_URL = os.getenv('WEB_HOOK_URL', '')
OUTPUT_RECORDE_SIZE = int(os.getenv('OUTPUT_RECORDE_SIZE', 10))
ACTIVEDATA_CHANNEL_ID = os.getenv('ACTIVEDATA_CHANNEL_ID', 0)
TARGET_DATE_DIFF = int(os.getenv('TARGET_DATE_DIFF', 7))
LATEST_MAX_COEFFICIENT = int(os.getenv('LATEST_MAX_COEFFICIENT', 5))
LATEST_MIN_COEFFICIENT = int(os.getenv('LATEST_MIN_COEFFICIENT', 1))

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
        ｔ,x = make_coefficient()
        ranking = make_ranking(df_user, df_article, t, x)

        #ファイル出力
        if len(ranking) == 0:
            output_file({'interest':interest, 'weakness':weakness})
        else:
            output_file({'interest':interest, 'weakness':weakness, 'ranking':ranking})
        post_slack('learn_job_result\nprocess_id:' + str(TIMESTAMP) + '\nresult:Success')
        print('End train handler.')
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        post_slack('learn_job_result\nprocess_id:' + str(TIMESTAMP) + '\nresult:Failure' + '\nerror:' + str(e))
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
        result = np.where(keyword["title"].isin(user_interest), 1*interest_coefficient, 1)
        result = np.append(result,np.where(skill["title"].isin(user_skill1), 1*weakness_coefficient, 1))
        result = np.append(result,np.where(role["title"].isin(user_role), 1, 0))
        result = np.append(result,np.where(skill["title"].isin(user_skill2), 1*weakness_coefficient, 1))
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

#内積の算出および内積の重み付きランダムサンプリング
def calculation_dot(user, article):
    data = []
    for u in user['result']:
        ret =[]
        for a in article['result']:
            v1 = np.array(u['vec'], dtype=float)
            v2 = np.array(a['vec'], dtype=float)
            ret.append({'id':a['id'], 'score':np.dot(v1, v2)})

        #内積結果による重みづけサンプリング
        ids = [d.get('id') for d in ret]
        scores = [d.get('score') for d in ret]
        np.random.seed()
        s = np.random.choice(ids, size=OUTPUT_RECORDE_SIZE, p=np.array(scores/sum(scores)), replace=False)
        s_list = s.tolist()

        #重みづけサンプリング後の並べ替え
        sample_list = []
        priority = 1
        for sl in s_list:
            for r in ret:
                if sl == r['id']:
                    sample_list.append({'id':r['id'], 'score':r['score'], 'priority':priority})
                    priority = priority + 1

        data.append({'personal': { 'id': u['id'] },'articles':sample_list})

    return data

#ランキングデータの作成
def make_ranking(user, article, t, x):
    try:
        #集計用配列
        click_cnt = []
        review_data = []

        #処理当日日付を取得(JST)
        t_delta = datetime.timedelta(hours=9)
        jst = datetime.timezone(t_delta, 'JST')

        #アクティブデータの取得
        datalake_client = DatalakeClient()
        channel = datalake_client.get_channel(ACTIVEDATA_CHANNEL_ID)
        files = channel.list_files(sort='-uploaded_at')

        for i, f in enumerate(files):
            file_item = channel.get_file(f.file_id)
            upload_day = dateutil.parser.parse(file_item.metadata['timestamp'])

            #ランキング対象データの日付判定
            if TARGET_DATE_DIFF >= (TODAY.date()- upload_day.date()).days:
                #zipファイルの解凍
                with zipfile.ZipFile(io.BytesIO(file_item.get_content())) as myzip:
                    fl = myzip.namelist()
                    idx = t.index(upload_day.date())
                    coefficient = x[idx]
                    for f in fl:
                        with myzip.open(f) as myfile:
                            if 'ClickData' in f:
                                data = pd.read_csv(myfile, encoding='utf-8', header=0)
                                #ID毎のクリック回数を集計
                                cnt = collections.Counter(data['article_id'].tolist())
                                #集計値に日付係数をかける
                                click_cnt.append(collections.Counter(list(cnt.elements()) * int(coefficient)))
                            elif 'ReviewData' in f:
                                data = pd.read_csv(myfile, encoding='utf-8', header=0)
                                #満足度を取得
                                if len(data) > 0:
                                    review_data.append(data[['article_id','satisfaction']])

        #ファイル単位のクリック回数を集計
        click_result = sum((collections.Counter(dict(x)) for x in click_cnt),collections.Counter())
        click_result = dict(click_result)

        #ファイル単位の満足度を結合し、IDごとの満足度の平均を集計
        review_result = pd.concat(review_data)
        review_result = review_result.groupby('article_id').mean().to_dict()['satisfaction']
        
        #データの作成
        result = []
        for aid in article.id:
            click_cnt = click_result.get(aid, 0)
            review_avg = review_result.get(aid, 1.0)
            result.append({'id': aid, 'score': click_cnt * review_avg })

        #スコアの降順にソート
        result = sorted(result, key=lambda x: (-x['score'],x['id']))

        #priority属性の付与
        priority = 1
        for r in result:
            r['priority'] = priority
            priority = priority + 1
        
        #ランキングデータ作成
        data=[]
        for uid in user.id:
            data.append({'personal': { 'id': uid },'articles':result[:OUTPUT_RECORDE_SIZE]})

        return data
    except Exception as e:
        post_slack('make_ranking_error'+'\nprocess_id:' + str(TIMESTAMP) + '\nresult:Failure' + '\nerror:' + str(e))
        return []

#ランキング係数用のデータ生成
def make_coefficient():
    # ランキング係数用の最大値、最小値
    target_date = TODAY + datetime.timedelta(days=-TARGET_DATE_DIFF)
    dt = datetime.timedelta(days=TARGET_DATE_DIFF)
    t = [target_date.date() + dt * x for x in range(2)]
    x = [LATEST_MIN_COEFFICIENT,LATEST_MAX_COEFFICIENT]

    # 補間用の日付データ作成
    dt_new = dt/TARGET_DATE_DIFF
    num_new = TARGET_DATE_DIFF + 1
    t_new = [t[0] + x * dt_new for x in range(num_new)]
    
    # 日付をdatetime型からunix時間（float）に変換する
    t_unix = [dateutil.parser.parse(x.strftime("%Y/%m/%d")).timestamp() for x in t]
    t_new_unix = [dateutil.parser.parse(x.strftime("%Y/%m/%d")).timestamp() for x in t_new]
    
    #線形補間
    x_new = np.interp(t_new_unix, t_unix, x)
    
    return t_new,x_new

#処理結果のSlack通知
def post_slack(message):
    requests.post(WEB_HOOK_URL, data = json.dumps({
        'text': message,  #通知内容
    }))

if __name__ == '__main__':
    handler(None)