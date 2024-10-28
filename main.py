import gradio as gr
import os
from dotenv import load_dotenv
import openai
from api_handlers import fetch_tourist_spots, fetch_restaurants_by_keyword
from database import init_db, check_db_contents

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def respond_to_query(query):
    """ユーザーからの問い合わせに応答"""
    try:
        # 観光地情報の取得
        tourist_info = fetch_tourist_spots(query)
        response = f"観光地の概要（{query}）:\n{tourist_info}\n\n"
        
        # レストラン情報の取得と整形
        restaurants = fetch_restaurants_by_keyword(query)
        if not restaurants:
            response += "レストラン情報が取得できませんでした。"
        else:
            response += "おすすめの飲食店:\n"
            for genre, shops in restaurants.items():
                if shops:
                    response += f"\n{genre}:\n" + "\n".join(
                        f"{i+1}. {r['name']} - {r.get('address', '住所情報なし')} "
                        f"({r.get('station', '駅情報なし')}) - TEL: {r['tel']} - "
                        f"予算: {r['budget']} - おすすめ: {r['recommended_dish']} - "
                        f"URL: {r['url']}"
                        for i, r in enumerate(shops)
                    )
                else:
                    response += f"\n{genre}: 該当する店舗が見つかりませんでした。"
        
        return response
    
    except Exception as e:
        return f"エラーが発生しました: {str(e)}\n\n正しい地名を入力してください。"

# Gradioインターフェースの設定
interface = gr.Interface(
    fn=respond_to_query,
    inputs=gr.Textbox(label="地名を入力してください"),
    outputs=gr.Textbox(label="観光情報"),
    title="東京観光情報ボット",
    description="東京の観光地名を入力すると、観光情報とおすすめ飲食店を表示します。"
)

if __name__ == "__main__":
    init_db()  # データベースの初期化
    # デバッグ用の確認
    from database import check_db_contents
    print("Database contents:", check_db_contents())
    interface.launch(share=True)
