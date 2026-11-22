import pickle
from collections import defaultdict
from datasets import load_dataset
from underthesea import word_tokenize
import time

print("Đang tải dữ liệu Wikipedia tiếng Việt (có thể mất 1-2 phút)...")
# Tải 10,000 bài viết đầu tiên từ Wikipedia tiếng Việt
dataset = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train[:10000]")

print("Đang xử lý văn bản...")
ngram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
total_articles = len(dataset)
start_time = time.time()

# Duyệt qua từng bài viết
for idx, article in enumerate(dataset):
    text = article['text'].lower()
    
    # (Tùy chọn) Tách từ tiếng Việt nếu bạn muốn Word-level N-gram
    # words = word_tokenize(text) 
    # Nếu chỉ làm Character-level (gõ từng chữ cái), ta dùng nguyên text
    
    # Quét từng ký tự để xây dựng Trigram (3 ký tự liên tiếp)
    # Ví dụ: "tôi đi" -> "t", "ô", "i", " ", "đ", "i"
    for i in range(len(text) - 2):
        c1, c2, c3 = text[i], text[i+1], text[i+2]
        # Bỏ qua các ký tự xuống dòng hoặc khoảng trắng kép
        if c1 == '\n' or c2 == '\n' or c3 == '\n': continue
        if c1 == ' ' and c2 == ' ': continue
        
        ngram_counts[c1][c2][c3] += 1
        
    # In tiến độ để bạn biết máy đang chạy
    if (idx + 1) % 2000 == 0:
        print(f"  Đã xử lý {idx + 1}/{total_articles} bài viết...")


print("Đang tính toán xác suất...")
# Sử dụng dict thường thay vì defaultdict để có thể pickle
ngram_probs = {}
for c1, next_chars in ngram_counts.items():
    ngram_probs[c1] = {}
    for c2, final_chars in next_chars.items():
        ngram_probs[c1][c2] = {}
        total_c1_c2 = sum(final_chars.values())
        for c3, count in final_chars.items():
            # Làm mịn (Smoothing)
            ngram_probs[c1][c2][c3] = (count + 1) / (total_c1_c2 + len(final_chars))

print("Đang lưu mô hình...")
with open('model_wikipedia.pkl', 'wb') as f:
    pickle.dump(ngram_probs, f)

print(f"Xong! Thời gian chạy: {time.time() - start_time:.2f} giây")
print("File model_wikipedia.pkl đã sẵn sàng.")