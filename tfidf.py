from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Dữ liệu ví dụ: một số câu tiếng Việt
documents = [
    "mèo thích ăn cá và chơi với chuột",
    "chó thích chạy và sủa to",
    "mèo và chó là bạn thân",
    "cá bơi trong nước rất nhanh"
]

# Khởi tạo TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Tính toán TF-IDF
tfidf_matrix = vectorizer.fit_transform(documents)

# Lấy danh sách các từ (features)
feature_names = vectorizer.get_feature_names_out()

# Chuyển ma trận TF-IDF thành DataFrame để dễ đọc
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# In kết quả
print("Ma trận TF-IDF:")
print(df_tfidf)

# In giá trị TF-IDF của từ "mèo" trong mỗi tài liệu
print("\nGiá trị TF-IDF của từ 'mèo':")
for i, doc in enumerate(documents):
    print(f"Tài liệu {i+1}: {doc} -> TF-IDF của 'mèo': {df_tfidf['mèo'][i]:.4f}")
