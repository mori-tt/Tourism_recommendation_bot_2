# 東京観光情報ボット

このプロジェクトは、ChatGPT API とホットペッパー API を利用して、東京の観光地や飲食店に関する情報を提供する BOT です。ユーザーが地名を入力すると、その地域の観光地の概要とおすすめの飲食店情報を取得できます。

## 機能

- **観光地の概要**: ChatGPT API を使用して、指定された地名の観光地の概要を 200 文字で提供します。([main.py:12-14](main.py))
- **飲食店情報**: ホットペッパー API を使用して、おすすめの飲食店（ラーメン、和食）の情報を提供します。表示される情報には店舗名、TEL、ジャンル、価格帯、おすすめ料理、URL が含まれます。([api_handlers.py:9-49](api_handlers.py))

## 未実装の機能

以下の機能は現在未実装ですが、将来的に追加する予定です。

### 安価でおすすめの宿泊施設情報

- **機能概要**: Google Maps API と Airbnb API を使用して、口コミが多いおすすめの宿泊施設トップ 5 を提供します。
- **表示情報**: 宿泊施設の名前、場所、概要、写真、予約リンク。

### 観光スポット情報

- **機能概要**: Google Maps API を使用して、口コミが多い観光スポットトップ 5 を提供します。
- **表示情報**: スポットの名前、場所、概要、写真。

### 文化財情報

- **機能概要**: Google Maps API と文化財一覧 API を使用して、地域の文化財情報を提供します。
- **表示情報**: 文化財の名前、場所、概要、写真、詳細内容。

### イベント情報

- **機能概要**: Google Maps API と東京都イベント一覧 API を使用して、入力された日から 2 週間以内のイベント情報トップ 5 を提供します。
- **表示情報**: イベントの名前、場所、概要、写真、詳細内容。

## 環境設定

このプロジェクトを実行する前に、以下の環境設定が必要です。

1. **Python のインストール**: Python 3.11.3 をインストールしてください。
2. **ライブラリのインストール**:
   必要な Python ライブラリをインストールします。

   pip install gradio openai requests python-dotenv

3. **環境変数の設定**:
   `.env` ファイルをプロジェクトのルートディレクトリに作成し、以下の内容を追加します。

   OPENAI_API_KEY=your_openai_api_key_here
   HOT_PEPPER_API=your_hot_pepper_api_key_here

## アプリケーションの起動

以下のコマンドを実行して、Gradio インターフェースを起動します。
python main.py

## 使用方法

Web ブラウザで表示された Gradio インターフェースにアクセスし、テキストボックスに地名を入力して「submit」ボタンを押すと、観光地の概要と飲食店情報が表示されます。

## トラブルシューティング

- **API キーが無効**: `.env`ファイルに正しい API キーが設定されていることを確認してください。
- **ライブラリが見つからない**: 必要なライブラリがすべてインストールされていることを確認し、必要に応じて再インストールしてください。
- **Gradio が起動しない**: コマンドプロンプトまたはターミナルでエラーメッセージを確認し、指示に従って問題を解決してください。

## アプリケーションの画面

### 入力前の画面

![入力前の画面](images/input.jpg)

### 入力後の画面

![入力後の画面](images/output.jpg)
# Tourism_recommendation_bot_2
