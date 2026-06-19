import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Nâng cấp: Kết hợp Tiêu đề và Tác giả để tăng độ sâu NLP
df = pd.read_csv('Books.csv', low_memory=False).drop_duplicates(subset='Book-Title')
df['metadata'] = df['Book-Title'].fillna('') + " " + df['Book-Author'].fillna('')
df = df.reset_index(drop=True)

# Vector hóa Metadata
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['metadata'])

def get_hybrid_ready_recommendations(book_name, top_n=20):
    if book_name not in df['Book-Title'].values:
        return None

    idx = df[df['Book-Title'] == book_name].index[0]

    # Tính tương quan Content-Based
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Ở ĐÂY: Trong tương lai, bạn sẽ cộng thêm Collaborative_Score vào đây
    # final_score = (alpha * content_score) + ((1-alpha) * collaborative_score)

    sim_indices = cosine_sim.argsort()[-(top_n+1):][::-1]
    sim_indices = sim_indices[1:] # Loại bỏ chính cuốn sách đó

    results = df.iloc[sim_indices][['Book-Title', 'Book-Author', 'Year-Of-Publication']].copy()
    results['Similarity-Score'] = cosine_sim[sim_indices] # Hiện điểm số để dễ debug
    results.index = range(1, len(results) + 1)
    return results

# Thực thi
book_query = 'Lord of The Ring'
recommendations = get_hybrid_ready_recommendations(book_query)

if recommendations is not None:
    print(f"Personalized Recommendations for '{book_query}':")
    display(recommendations)
else:
    print(f"No recommendations found for '{book_query}'.")