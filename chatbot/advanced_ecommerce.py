import json
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any
import os
from torch.utils.data import DataLoader

def fine_tune_model(model_name='all-MiniLM-L6-v2', output_path='fine_tuned_ecommerce_model', epochs=1, batch_size=16):
    """
    Tinh chỉnh mô hình SentenceTransformer trên dữ liệu e-commerce.
    """
    # Kiểm tra file dữ liệu
    json_file_path = 'ecommerce_knowledgebase.json'
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} không tồn tại! Không thể tinh chỉnh mô hình.")
        return False

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
    except Exception as e:
        print(f"Error: Không thể đọc file {json_file_path} - {e}")
        return False

    # Nhóm các câu hỏi theo danh mục
    category_questions = {}
    for item in knowledge_data:
        cat = item.get('category', 'general')
        if cat not in category_questions:
            category_questions[cat] = []
        category_questions[cat].append(item['question'])

    # Tạo cặp positive (các câu hỏi trong cùng danh mục)
    positive_pairs = []
    for questions in category_questions.values():
        if len(questions) > 1:
            for i in range(len(questions)):
                for j in range(i + 1, len(questions)):
                    positive_pairs.append((questions[i], questions[j]))

    if not positive_pairs:
        print("Warning: Không đủ dữ liệu để tinh chỉnh (cần ít nhất 2 câu hỏi trong cùng danh mục).")
        return False

    # Tạo dữ liệu huấn luyện
    train_examples = [InputExample(texts=[pair[0], pair[1]]) for pair in positive_pairs]
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Tinh chỉnh mô hình
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path
        )
        print(f"Mô hình đã được tinh chỉnh và lưu tại {output_path}")
        return True
    except Exception as e:
        print(f"Error: Lỗi khi tinh chỉnh mô hình - {e}")
        return False

if __name__ == "__main__":
    model_path = 'fine_tuned_ecommerce_model'
    if not os.path.exists(model_path):
        print("Mô hình tinh chỉnh chưa tồn tại. Đang tiến hành tinh chỉnh...")
        success = fine_tune_model(epochs=1)
        if not success:
            print("Không thể tinh chỉnh mô hình. Sử dụng mô hình mặc định 'all-MiniLM-L6-v2'.")
            model_path = 'all-MiniLM-L6-v2'
    else:
        print(f"Sử dụng mô hình đã tinh chỉnh: {model_path}")
