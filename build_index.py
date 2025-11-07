from retrieve_utils import retrive_news, build_index
from langchain_ollama.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

sources = [
    "https://www.bbc.com/news",
    "https://edition.cnn.com",
    "https://www.thehindu.com",
    "https://techcrunch.com",
    "https://indianexpress.com",
    "https://www.aljazeera.com/",
    "https://www.livemint.com/news",
    "https://www.thequint.com/",
    "https://www.morningbrew.com/",
    "https://asia.nikkei.com/",
    "https://www.marketwatch.com/",
    "https://www.nytimes.com/international/",

    "https://ew.com/",
    "https://techcrunch.com/latest/",
    "https://arstechnica.com/",
    "https://www.wired.com/category/science/",
    "https://www.theverge.com/",
    "https://variety.com/",
    "https://www.reuters.com/world/",
    "https://www.theguardian.com/world",
    "https://pitchfork.com/"
]

news_df = retrive_news(sources=sources,
                       max_articles=40)

# Saving CSV
print("-"*25, "\n")
print("Saving articles to CSV...")
news_df.to_csv("articles.csv",
               index=False,
               sep=";")

# Buildig index
print("-"*25, "\n")
print("Building Index...")
index = build_index(df=news_df,
                    embeddings=embeddings)
index.save_local("news_index")

print("\nDone.")