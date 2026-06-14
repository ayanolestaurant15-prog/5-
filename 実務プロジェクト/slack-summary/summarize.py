# -*- coding: utf-8 -*-
"""
Slackチャンネルのメッセージを取得して、LLM(OpenAI API)で要約し、
その要約をSlackに投稿し、さらにLINEにも送信するプログラム。

【処理の流れ】
  1. Slack APIで指定チャンネルの過去24時間のメッセージを取得する
  2. ユーザーIDを表示名(名前)に変換して、読みやすいログに整形する
  3. prompt.txt のテンプレートにログを埋め込み、OpenAI APIに送って要約してもらう
  4. できあがった要約をSlackチャンネルに投稿する
  5. 同じ要約をLINE Messaging APIで指定の宛先に送信する

実行方法:  python summarize.py
"""

import os
from datetime import datetime, timedelta

import requests                         # LINEのAPIを呼ぶためのHTTP通信ライブラリ
from dotenv import load_dotenv          # .envファイルから設定を読み込むためのライブラリ
from slack_sdk import WebClient         # Slack APIを簡単に使うための公式ライブラリ
from slack_sdk.errors import SlackApiError
from openai import OpenAI               # OpenAI(ChatGPT)のAPIを使うためのライブラリ

# --------------------------------------------------
# 設定の読み込み
# --------------------------------------------------
# .envファイルに書いたトークンなどを環境変数として読み込む。
# トークンは「パスワード」のようなものなので、プログラムに直接書かず
# .envファイルに分けておくのが安全な書き方。
load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]    # Slackボットのトークン (xoxb-で始まる)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]      # OpenAIのAPIキー (sk-で始まる)
CHANNEL_ID = os.environ["SLACK_CHANNEL_ID"]        # 要約したいチャンネルのID (Cで始まる)

# LINE関連の設定。未設定でもSlackへの投稿だけは動くよう .get で読み込む
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")  # LINEのチャネルアクセストークン
LINE_TO_ID = os.environ.get("LINE_TO_ID")          # 送信先のユーザーID or グループID

HOURS_BACK = 24          # 何時間前までのメッセージを取得するか
OPENAI_MODEL = "gpt-4o-mini"   # 使うAIモデル。安くて速いので練習用におすすめ

slack = WebClient(token=SLACK_BOT_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def fetch_messages(channel_id: str, hours_back: int) -> list:
    """指定チャンネルから、直近 hours_back 時間分のメッセージを取得する。"""
    # Slackは時刻を「UNIXタイムスタンプ」(1970年からの経過秒数)で扱うので変換する
    oldest = (datetime.now() - timedelta(hours=hours_back)).timestamp()

    messages = []
    cursor = None  # メッセージが多いとき、続きを取得するための「しおり」

    while True:
        response = slack.conversations_history(
            channel=channel_id,
            oldest=str(oldest),
            limit=200,
            cursor=cursor,
        )
        messages.extend(response["messages"])

        # next_cursorが空なら全部取得できたのでループを抜ける
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    # Slackは新しい順で返してくるので、読みやすいように古い順に並べ替える
    messages.reverse()
    return messages


def get_user_name(user_id: str, cache: dict) -> str:
    """ユーザーID(例: U123ABC)を表示名(例: 田中)に変換する。

    同じ人を何度もAPIに問い合わせると遅いので、
    一度調べた名前はcache(辞書)に保存して使い回す。
    """
    if user_id in cache:
        return cache[user_id]
    try:
        info = slack.users_info(user=user_id)
        profile = info["user"]["profile"]
        # 表示名が未設定の人もいるので、その場合は本名を使う
        name = profile.get("display_name") or profile.get("real_name") or user_id
    except SlackApiError:
        name = user_id
    cache[user_id] = name
    return name


def format_messages(messages: list) -> str:
    """メッセージのリストを「[時刻] 名前: 本文」形式のテキストに整形する。"""
    name_cache = {}
    lines = []

    for msg in messages:
        # ボットの投稿や「チャンネルに参加しました」などのシステム通知は除外する
        if msg.get("subtype") or "user" not in msg:
            continue

        text = msg.get("text", "").strip()
        if not text:
            continue

        time_str = datetime.fromtimestamp(float(msg["ts"])).strftime("%m/%d %H:%M")
        name = get_user_name(msg["user"], name_cache)
        lines.append(f"[{time_str}] {name}: {text}")

    return "\n".join(lines)


def summarize(message_log: str, channel_name: str, date_range: str) -> str:
    """整形したメッセージログをOpenAI APIに送り、要約文を返してもらう。"""
    # prompt.txt を読み込んで、{messages} などの部分を実際の値に置き換える
    prompt_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(prompt_dir, "prompt.txt"), encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        channel_name=channel_name,
        date_range=date_range,
        messages=message_log,
    )

    # OpenAI APIの呼び出し。
    # ChatGPTの画面で質問を送るのと同じことを、プログラムから行っている。
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # 0に近いほど毎回安定した答えになる(要約向き)
    )
    return response.choices[0].message.content


def send_to_line(text: str) -> None:
    """要約文をLINE Messaging APIで指定の宛先にプッシュ送信する。

    LINEの「プッシュメッセージ」は、こちらから好きなタイミングで
    メッセージを送る仕組み。指定の宛先(自分やグループ)に届く。
    """
    # トークンか宛先が未設定なら、LINE送信はスキップする(Slack投稿だけで終わる)
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_TO_ID:
        print("LINEの設定(.env)が未入力のため、LINE送信をスキップしました。")
        return

    # LINEのプッシュメッセージ用の窓口(エンドポイント)
    url = "https://api.line.me/v2/bot/message/push"

    # 「合鍵(トークン)」を付けて、自分が正規の利用者であることを示す
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    # 送るデータ。to=宛先、messages=送る中身(今回はテキスト1件)
    payload = {
        "to": LINE_TO_ID,
        "messages": [{"type": "text", "text": text}],
    }

    response = requests.post(url, headers=headers, json=payload)

    # ステータスコード200なら成功。それ以外はエラー内容を表示する
    if response.status_code == 200:
        print("LINEへの送信に成功しました。")
    else:
        print(f"LINEへの送信に失敗しました (コード: {response.status_code})")
        print(response.text)


def main():
    # チャンネル名を取得(要約の見出しに使う)
    channel_info = slack.conversations_info(channel=CHANNEL_ID)
    channel_name = "#" + channel_info["channel"]["name"]

    print(f"{channel_name} から過去{HOURS_BACK}時間のメッセージを取得中...")
    messages = fetch_messages(CHANNEL_ID, HOURS_BACK)
    message_log = format_messages(messages)

    if not message_log:
        print("対象期間にメッセージがなかったため、終了します。")
        return

    print(f"{len(message_log.splitlines())}件のメッセージを要約中...")
    date_range = (
        f"{(datetime.now() - timedelta(hours=HOURS_BACK)).strftime('%m/%d %H:%M')}"
        f" 〜 {datetime.now().strftime('%m/%d %H:%M')}"
    )
    summary = summarize(message_log, channel_name, date_range)

    print("要約をSlackに投稿中...")
    slack.chat_postMessage(channel=CHANNEL_ID, text=summary)

    print("要約をLINEに送信中...")
    send_to_line(summary)

    print("完了しました!")
    print("-" * 40)
    print(summary)


if __name__ == "__main__":
    main()
