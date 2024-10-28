import os
from dotenv import load_dotenv
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from database import get_tourist_spot
from langchain.schema import Document

load_dotenv()

def fetch_tourist_spots(area_name):
    try:
        # データベースから観光地情報を取得（RAGの參
        tourist_info = get_tourist_spot(area_name)
        
        # Document形式で情報を作成
        documents = []
        if tourist_info:
            documents.append(Document(page_content=tourist_info, metadata={"source": "database"}))
        
        # Web情報の取得を試みる
        try:
            web_urls = [
                f"https://www.gotokyo.org/jp/destinations/{area_name}/index.html",
                f"https://www.gotokyo.org/jp/spot/{area_name}/index.html",
                f"https://www.gotokyo.org/jp/areas/{area_name}.html"
            ]
            for url in web_urls:
                try:
                    web_documents = WebBaseLoader([url]).load()
                    documents.extend(web_documents)
                except Exception:
                    continue
        except Exception:
            pass
        
        # データベースにもWebにも情報がない場合は、AIに地名の存在確認を依頼
        if not documents:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            existence_check = llm.invoke(
                f"「{area_name}」は東京都内の実在する地名ですか？単に'yes'か'no'で答えてください。"
            )
            
            if "yes" in existence_check.content.lower():
                default_info = (
                    f"{area_name}は東京都内の実在する地域です。"
                    "以下の情報はAIの一般的な知識に基づいて生成します。"
                )
                documents.append(Document(page_content=default_info, metadata={"source": "ai"}))
            else:
                return f"{area_name}の観光情報が見つかりませんでした。"
        
        # テキスト分割の設定
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())
        
        # ChatGPTによる情報生成
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        prompt = ChatPromptTemplate.from_template("""
入力された{area_name}についてのみ説明してください。
他のエリアへの言及は避け、その地域に実在する観光スポットと特徴のみ説明してください。

参考情報：
{context}

以下の形式で回答を作成してください：

【観光スポットと特徴】
（その地域の実在する観光スポット、特徴、雰囲気について500文字程度で説明）

【観光シーズンとイベント】
（その地域で実際に開催されているイベントや、最適な観光シーズンについて100文字程度で説明）

注意事項：
1. 他の地域への言及は避けてください
2. 確実な情報のみを含めてください
3. 世界遺産や重要文化財がある場合のみ、その情報を含めてください
4. 不確かな情報は含めないでください
""")
        
        chain = (
            {"context": RunnablePassthrough(), "area_name": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        docs = vectorstore.similarity_search(area_name, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        response = chain.invoke({"area_name": area_name, "context": context})
        return response.content

    except Exception as e:
        print(f"Error in fetch_tourist_spots: {e}")
        return f"{area_name}の観光情報を取得できませんでした。"

def fetch_restaurants_by_keyword(area_name):
    """ホットペッパーAPIを使用してレストラン情報を取得"""
    api_key = os.getenv("HOT_PEPPER_API")
    if not api_key:
        return None
    
    base_url = "http://webservice.recruit.co.jp/hotpepper/gourmet/v1/"
    
    all_restaurants = {
        "レストラン": [],
        "ラーメン": []
    }
    
    # 基本パラメータ
    params = {
        "key": api_key,
        "format": "json",
        "count": 5,
        "large_area": "Z011",  # 東京
        "keyword": area_name
    }
    
    # 駅名検索用のパラメータ
    station_params = params.copy()
    station_params["station_name"] = area_name
    
    # 住所検索のパラメータ
    address_params = params.copy()
    address_params["address"] = area_name
    
    try:
        for search_params in [params, station_params, address_params]:
            # 一般レストラン
            response = requests.get(base_url, params=search_params)
            data = response.json()
            if "results" in data and "shop" in data["results"]:
                all_restaurants["レストラン"].extend([{
                    "name": r["name"],
                    "tel": r.get("tel", "情報なし"),
                    "budget": r.get("budget", {}).get("name", "情報なし"),
                    "recommended_dish": r.get("genre", {}).get("catch", "情報なし"),
                    "address": r.get("address", "情報なし"),
                    "station": r.get("station_name", "情報なし"),
                    "url": r["urls"]["pc"]
                } for r in data["results"]["shop"]])
            
            # ラーメン店
            ramen_params = search_params.copy()
            ramen_params["genre"] = "G013"
            ramen_response = requests.get(base_url, params=ramen_params)
            ramen_data = ramen_response.json()
            if "results" in ramen_data and "shop" in ramen_data["results"]:
                all_restaurants["ラーメン"].extend([{
                    "name": r["name"],
                    "tel": r.get("tel", "情報なし"),
                    "budget": r.get("budget", {}).get("name", "情報なし"),
                    "recommended_dish": r.get("genre", {}).get("catch", "情報なし"),
                    "address": r.get("address", "情報なし"),
                    "station": r.get("station_name", "情報なし"),
                    "url": r["urls"]["pc"]
                } for r in ramen_data["results"]["shop"]])
        
        # 重複除去
        for genre in all_restaurants:
            seen = set()
            all_restaurants[genre] = [
                x for x in all_restaurants[genre] 
                if x["name"] not in seen and not seen.add(x["name"])
            ][:3]
        
        return all_restaurants if any(all_restaurants.values()) else None
        
    except Exception as e:
        print(f"Error fetching restaurants: {e}")
        return None
