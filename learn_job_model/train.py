from abeja.datalake import Client as DatalakeClient
import abeja.datalake as aj
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
import random
from typing import Tuple,Dict,Union

# 出力ディレクトリの作成
# ABEJA_TRAINING_RESULT_DIRという環境変数に出力先ディレクトリが設定される
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', '.')
os.makedirs(ABEJA_TRAINING_RESULT_DIR, exist_ok=True)

#環境変数の取り込み
INPUT_CHANNEL_ID = os.getenv('INPUT_CHANNEL_ID', 0)
OUTPUT_CHANNEL_ID = os.getenv('OUTPUT_CHANNEL_ID', 0)
INTEREST_COEFFICIENT = int(os.getenv('INTEREST_COEFFICIENT', 40))
WEAKNESS_COEFFICIENT = int(os.getenv('WEAKNESS_COEFFICIENT', 2))
WEB_HOOK_URL = os.getenv('WEB_HOOK_URL', '')
OUTPUT_RECORDE_SIZE = int(os.getenv('OUTPUT_RECORDE_SIZE', 20))
ACTIVEDATA_CHANNEL_ID = os.getenv('ACTIVEDATA_CHANNEL_ID', 0)
TARGET_DATE = int(os.getenv('TARGET_DATE', 0))
TARGET_DATE_DIFF = int(os.getenv('TARGET_DATE_DIFF', 14))
LATEST_MAX_COEFFICIENT = int(os.getenv('LATEST_MAX_COEFFICIENT', 5))
LATEST_MIN_COEFFICIENT = int(os.getenv('LATEST_MIN_COEFFICIENT', 1))
MOVIE_SETTING = os.getenv('MOVIE_SETTING', '1,4')

#処理当日日付を取得(JST)
t_delta = datetime.timedelta(hours=9)
jst = datetime.timezone(t_delta, 'JST')
TODAY = datetime.datetime.now(jst) + datetime.timedelta(days=TARGET_DATE)

#処理IDの取得
TIMESTAMP = int(time.time() * 1000)

def handler(context):
    print('Start train handler.')
    print(ABEJA_TRAINING_RESULT_DIR)
    try:

        #マスタファイルの読み込み
        df_user, df_article_all, df_keyword, df_role, df_skill, df_level = input_datalake_master()

        #記事マスタをテキストと動画に分類
        df_article = df_article_all[df_article_all.original != 'movie']
        df_movie = df_article_all[df_article_all.original == 'movie']

        #動画設定情報ををintの配列に置換
        movie_setting = [int(s) for s in MOVIE_SETTING.split(',')]

        #interestデータ作成
        user_vec = usertovec(df_user, df_keyword, df_role, df_skill, df_level, INTEREST_COEFFICIENT*2, WEAKNESS_COEFFICIENT)
        article_vec = articletovec(df_article, df_keyword, df_role, df_skill, df_level)
        movie_vec = articletovec(df_movie, df_keyword, df_role, df_skill, df_level)
        #interestスコアの高い動画情報の取得
        movie_info = get_movie(user_vec, movie_vec, len(movie_setting))
        #内積処理
        interest = calculation_dot(user_vec, article_vec, movie_info, movie_setting)

        #weaknessデータ作成
        user_vec = usertovec(df_user, df_keyword, df_role, df_skill, df_level, WEAKNESS_COEFFICIENT, INTEREST_COEFFICIENT)
        #weaknessスコアの高い動画情報の取得
        movie_info = get_movie(user_vec, movie_vec, len(movie_setting))
        #内積処理
        weakness = calculation_dot(user_vec, article_vec, movie_info, movie_setting)

        #rankingデータ生成
        t,x = make_coefficient()
        ranking = make_ranking(df_user, df_article_all, t, x)

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
def input_datalake_master() -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame, pd.core.frame.DataFrame
                                   , pd.core.frame.DataFrame, pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
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

#ファイル出力
def output_file(obj: Dict[str, dict]) -> None:
    for k, v in obj.items():
        metadata = { 'filename': str(TIMESTAMP) + '_' + k + '.json' }
        d  = {'timestamp':TIMESTAMP, 'recommend_type':k, 'result':v}
        output_datalake(d, metadata)
        output_dir(k, v)

#ディレクトリへの出力
def output_dir(recommend_type: str, obj: dict) -> None:
    with open(os.path.join(ABEJA_TRAINING_RESULT_DIR, recommend_type + '.json'), 'w') as f:
        json.dump(obj, f, default=expireEncoda, ensure_ascii=False)

#指定のデータレイクチャンネルへのファイル出力
def output_datalake(object: dict, metadata: dict) -> None:
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(OUTPUT_CHANNEL_ID)
    channel.upload(json.dumps(object, default=expireEncoda, ensure_ascii=False).encode('utf-8'), metadata=metadata, content_type='application/json')

#オブジェクトのエンコード
def expireEncoda(object):
    if isinstance(object, np.integer):
        return int(object)

#ユーザーのベクトル化
def usertovec(
    user: pd.core.frame.DataFrame,
    keyword: pd.core.frame.DataFrame,
    role: pd.core.frame.DataFrame,
    skill: pd.core.frame.DataFrame,
    level: pd.core.frame.DataFrame,
    interest_coefficient: Union[int,float],
    weakness_coefficient: Union[int,float]
) -> dict:
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

        #ベクトル作成（ユーザー特徴量）
        #ユーザーマスタの興味とキーワードマスタの興味が合致した場合に、ユーザーの特徴量に重みづけを行う
        result = np.where(keyword['title'].isin(user_interest), interest_coefficient, 1)
        #ユーザーマスタのスキルとスキルマスタのスキルが合致した場合に、ユーザーの特徴量に重みづけを行う
        result = np.append(result,np.where(skill['title'].isin(user_skill1), weakness_coefficient, 1))
        #ユーザーマスタのロールとロールマスタの役割が合致した場合に、ユーザーの特徴量に重みづけを行う
        result = np.append(result,np.where(role['title'].isin(user_role), 1, 0))
        #ユーザーマスタの役割に紐づくスキルとスキルマスタのスキルが合致した場合に、ユーザーの特徴量に重みづけを行う
        result = np.append(result,np.where(skill['title'].isin(user_skill2), weakness_coefficient, 0))
        #ユーザーマスタのレベルとレベルマスタのレベルが合致した場合に、ユーザーの特徴量に重みづけを行う
        result = np.append(result,np.where(level['title'].isin(user_level), 1, 0))
        dict_obj = {'id':data.id, 'vec':result.tolist()}
        user_vec['result'].append(dict_obj)
    return user_vec

#コンテンツのベクトル化
def articletovec(
    article: pd.core.frame.DataFrame,
    keyword: pd.core.frame.DataFrame,
    role: pd.core.frame.DataFrame,
    skill: pd.core.frame.DataFrame,
    level: pd.core.frame.DataFrame
) -> dict:
    article_vec = {'result':[]}
    for data in article.itertuples():
        #興味関心
        art_interest = [data.interest1, data.interest2, data.interest3]
        #スキル
        art_skill = [data.skill1, data.skill2, data.skill3]
        #スキルを含む役割
        conditions = {'skill1': art_skill, 'skill2': art_skill, 'skill3':art_skill, 'skill4':art_skill}
        r = role.loc[lambda x:(x[conditions.keys()].isin(conditions).any(axis=1))]
        art_role = [v for v in r['title'].values]
        #レベル
        art_level = [data.level]

        #ベクトル作成（コンテンツ特徴量）
        #コンテンツマスタの興味とキーワードマスタの興味が合致した場合に、コンテンツの特徴量に重みづけを行う
        result = np.where(keyword['title'].isin(art_interest), 1, 0)
        #コンテンツマスタのスキルとスキルマスタのスキルが合致した場合に、コンテンツの特徴量に重みづけを行う
        result = np.append(result,np.where(skill['title'].isin(art_skill), 1, 0))
        #ユーザーマスタのスキルが必要なロールとロールマスタの役割が合致した場合に、コンテンツの特徴量に重みづけを行う
        result = np.append(result,np.where(role['title'].isin(art_role), 1, 0))
        #コンテンツマスタのスキルとスキルマスタのスキルが合致した場合に、コンテンツの特徴量に重みづけを行う
        result = np.append(result,np.where(skill['title'].isin(art_skill), 1, 0))
        #コンテンツマスタのレベルとレベルマスタのレベルが合致した場合に、コンテンツの特徴量に重みづけを行う
        result = np.append(result,np.where(level['title'].isin(art_level), 1, 0))
        dict_obj = {'id':data.id, 'vec':result.tolist()}
        article_vec['result'].append(dict_obj)
    return article_vec

#内積の算出および内積の重み付きランダムサンプリング
def calculation_dot(user: dict, article: dict, movie: dict, movie_stting: list) -> dict:
    data = []
    for u in user['result']:
        ret = [{'id':a['id'], 'score':np.dot(np.array(u['vec'], dtype=float), np.array(a['vec'], dtype=float))} for a in article['result']]

        #内積結果による重みづけサンプリング
        ids = [d.get('id') for d in ret]
        scores = [d.get('score') for d in ret]
        np.random.seed()
        s = np.random.choice(ids, size=OUTPUT_RECORDE_SIZE, p=np.array(scores/sum(scores)), replace=False)
        s_list = s.tolist()

        #動画コンテンツの設定
        #ユーザーに該当する動画情報の取得
        movie_list = list(filter(lambda x : x['uid'] == u['id'], movie))
        movie_list = movie_list[0]
        #設定順位に動画コンテンツの設定
        idx = 0
        for m_idx in movie_stting:
            if len(movie_list['movie']) > idx:
                s_list.insert(m_idx-1, movie_list['movie'][idx]['id'])
                idx += 1

        ret = ret + movie_list['movie']

        #重みづけサンプリング後の並べ替え
        sample_list = []
        priority = 1
        for sl in s_list:
            score = [r.get('score') for r in ret if r.get('id') == sl]
            sample_list.append({'id':sl, 'score':score[0], 'priority':priority})
            priority += 1

        data.append({'personal': { 'id': u['id'] },'articles':sample_list[:OUTPUT_RECORDE_SIZE]})

    return data

#スコアの高い動画コンテンツの抽出
def get_movie(user: dict, movie: dict, add_size:int) -> dict:
    data = []
    for u in user['result']:

        ret =[{'id':a['id'], 'score':np.dot(np.array(u['vec'], dtype=float), np.array(a['vec'], dtype=float))} for a in movie['result']]

        #scoreが係数以上（マッチしている動画）を取り出し、設定値分だけランダムで抽出
        ret = list(filter(lambda x : x['score'] > INTEREST_COEFFICIENT , ret))
        size = min(add_size, len(ret))
        ret = random.sample(ret, size)

        data.append({'uid': u['id'] ,'movie':ret})

    return data

#ランキングデータの作成
def make_ranking(
    user: pd.core.frame.DataFrame,
    article: pd.core.frame.DataFrame,
    t: list,
    x: np.ndarray
) -> dict:
    try:
        #集計用配列
        click_cnt = []
        review_data = []

        #アクティブデータの取得
        datalake_client = DatalakeClient()
        channel = datalake_client.get_channel(ACTIVEDATA_CHANNEL_ID)
        #対象日付クエリパラメータ作成
        target_date = TODAY + datetime.timedelta(days=-TARGET_DATE_DIFF)
        _query = 'x-abeja-meta-timestamp:>=' + target_date.strftime('%Y-%m-%d') + ' AND x-abeja-meta-timestamp:<=' + TODAY.strftime('%Y-%m-%dT%H:%M:%S')
        files = channel.list_files(sort='-uploaded_at',query=_query)

        for i, f in enumerate(files):
            file_item = channel.get_file(f.file_id)
            #zipファイルを解答し、クリックデータとレビューデータを取得
            click ,review = load_zip(file_item, t, x)
            
            click_cnt.append(click)
            if len(review) > 0:
                review_data.append(review)

        #ファイル単位のクリック回数を集計
        click_result = sum((collections.Counter(dict(x)) for x in click_cnt),collections.Counter())
        click_result = dict(click_result)

        #ファイル単位の満足度を結合し、IDごとの満足度の平均を集計
        if len(review_data) > 0:
            review_result = pd.concat(review_data)
            review_result = review_result.groupby('article_id').mean().to_dict()['satisfaction']
        else:
            review_result = {}

        #データの作成
        result = [{'id': aid, 'score': click_result.get(aid, 0) * review_result.get(aid, 1.0)} for aid in article.id]

        #スコアの降順にソート
        result = sorted(result, key=lambda x: (-x['score']))

        #priority属性の付与
        priority = 1
        for r in result:
            r['priority'] = priority
            priority += 1
        
        #ランキングデータ作成
        data = [{'personal': { 'id': uid },'articles':result[:OUTPUT_RECORDE_SIZE]} for uid in user.id]

        return data
    except Exception as e:
        post_slack('make_ranking_error'+'\nprocess_id:' + str(TIMESTAMP) + '\nresult:Failure' + '\nerror:' + str(e))
        return []

#zipファイルを読み込んで、クリックデータとレビューデータを取得
def load_zip(
    file_item: aj.file.DatalakeFile,
    t: list,
    x: np.ndarray
)  -> Tuple[collections.Counter, pd.core.frame.DataFrame]:

    upload_day = dateutil.parser.parse(file_item.metadata['timestamp'])
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
                    _click = collections.Counter(list(cnt.elements()) * int(coefficient))
                elif 'ReviewData' in f:
                    data = pd.read_csv(myfile, encoding='utf-8', header=0)
                    #満足度を取得
                    _review = data[['article_id','satisfaction']]
    return _click, _review

#ランキング係数用のデータ生成
def make_coefficient() -> Tuple[list, np.ndarray]:

    if TARGET_DATE_DIFF == 0:
        #一日分の実行の場合は、処理日のレコードのみ返却
        return [TODAY.date()],[LATEST_MAX_COEFFICIENT]
    else:
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
        t_unix = [dateutil.parser.parse(x.strftime('%Y/%m/%d')).timestamp() for x in t]
        t_new_unix = [dateutil.parser.parse(x.strftime('%Y/%m/%d')).timestamp() for x in t_new]
        
        #線形補間
        x_new = np.interp(t_new_unix, t_unix, x)
        
        return t_new,x_new

#処理結果のSlack通知
def post_slack(message: str) -> None:
    requests.post(WEB_HOOK_URL, data = json.dumps({
        'text': message,  #通知内容
    }))

if __name__ == '__main__':
    handler(None)