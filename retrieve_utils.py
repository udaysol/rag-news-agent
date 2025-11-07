import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import pandas as pd
import nltk

from newspaper import build
from tqdm import tqdm

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
# nltk.download('punkt_tab')

# Retrieve News from various sources
def retrive_news(sources: list, max_articles: int) -> pd.DataFrame:
    all_articles = []

    for source in sources:
        print(f"\n📰 Fetching from: {source}")
        paper = build(source, memoize_articles=False)

        for content in tqdm(paper.articles[:max_articles]):
            try:
                content.download()
                content.parse()
                content.nlp()  
                all_articles.append({
                    "source": source,
                    "title": content.title,
                    "authors": ", ".join(content.authors),
                    "publish_date": content.publish_date,
                    "url": content.url,
                    "summary": content.summary,
                    "keywords": ", ".join(content.keywords),
                    "text": content.text
                })
            except Exception as e:
                print(f"❌ Skipped: {e}")
                continue

    return pd.DataFrame(all_articles)

# Building FAISS index
def build_index(df: pd.DataFrame, embeddings: OllamaEmbeddings) -> FAISS:
    df["summary"] = df["summary"].fillna("").astype(str).str.strip()
    df = df[df["summary"] != ""]
    texts = df["summary"].tolist()
    meta = df[["title", "source", "url"]].to_dict(orient="records")

    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, meta)]
    db = FAISS.from_documents(documents=docs,
                              embedding=embeddings)

    return db
