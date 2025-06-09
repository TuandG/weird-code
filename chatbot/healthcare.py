# Healthcare Chatbot Knowledge Base
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime
import re
import os

class HealthcareKnowledgeBase:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Khởi tạo Knowledge Base cho Healthcare Chatbot
        """
        self.model = SentenceTransformer(model_name)
        self.knowledge_data = []
        self.embeddings = []
        self.db_path = "healthcare_kb.db"
        self.init_database()
        self.load_healthcare_data()
    
    def init_database(self):
        """Khởi tạo database SQLite"""
        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        
        # Tạo bảng knowledge base
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                question TEXT,
                answer TEXT,
                keywords TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tạo bảng chat history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_query TEXT,
                bot_response TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_healthcare_data(self):
        """Load dữ liệu healthcare từ file JSON vào knowledge base"""
        try:
            # Đường dẫn tới file JSON
            json_file_path = "healthcare_knowledgebase.json"
        
            # Kiểm tra file có tồn tại không
            if not os.path.exists(json_file_path):
                print(f"Warning: File {json_file_path} không tồn tại!")
                return []
        
            # Đọc dữ liệu từ file JSON
            with open(json_file_path, 'r', encoding='utf-8') as file:
                healthcare_data = json.load(file)
        
            print(f"Đã load thành công {len(healthcare_data)} câu hỏi từ {json_file_path}")
        
        except json.JSONDecodeError as e:
            print(f"Error: Lỗi format JSON - {e}")
        except FileNotFoundError:
            print(f"Error: Không tìm thấy file {json_file_path}")
        except Exception as e:
            print(f"Error: Lỗi không xác định - {e}")
        
        # Lưu vào database và tạo embeddings
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in healthcare_data:
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_base 
                (category, question, answer, keywords, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (item['category'], item['question'], item['answer'], 
                  item['keywords'], 1.0))
            
            self.knowledge_data.append(item)
        
        conn.commit()
        conn.close()
        
        # Tạo embeddings cho tất cả câu hỏi
        questions = [item['question'] + " " + item['keywords'] for item in self.knowledge_data]
        self.embeddings = self.model.encode(questions)
    
    def find_best_answer(self, user_query, threshold=0.5):
        """Tìm câu trả lời phù hợp nhất cho câu hỏi"""
        # Tạo embedding cho câu hỏi của user
        query_embedding = self.model.encode([user_query])
        
        # Tính cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Tìm câu trả lời có độ tương đồng cao nhất
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            answer = self.knowledge_data[best_idx]['answer']
            category = self.knowledge_data[best_idx]['category']
            
            # Lưu lịch sử chat
            self.save_chat_history(user_query, answer, best_score)
            
            return {
                "answer": answer,

                "confidence": float(best_score),
                "category": category,
                "source": "knowledge_base"
            }
        else:
            fallback_answer = "Tôi chưa có thông tin chính xác về vấn đề này. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để được tư vấn tốt nhất."
            self.save_chat_history(user_query, fallback_answer, best_score)
            
            return {
                "answer": fallback_answer,
                "confidence": float(best_score),
                "category": "unknown",
                "source": "fallback"
            }
    
    def save_chat_history(self, query, response, confidence):
        """Lưu lịch sử chat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_history (user_query, bot_response, confidence)
            VALUES (?, ?, ?)
        ''', (query, response, confidence))
        conn.commit()
        conn.close()
    
    def add_knowledge(self, category, question, answer, keywords):
        """Thêm kiến thức mới vào knowledge base"""
        # Lưu vào database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO knowledge_base (category, question, answer, keywords, confidence_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (category, question, answer, keywords, 1.0))
        conn.commit()
        conn.close()
        
        # Cập nhật knowledge data và embeddings
        new_item = {
            "category": category,
            "question": question,
            "answer": answer,
            "keywords": keywords
        }
        self.knowledge_data.append(new_item)
        
        # Tạo lại embeddings
        questions = [item['question'] + " " + item['keywords'] for item in self.knowledge_data]
        self.embeddings = self.model.encode(questions)
    
    def get_chat_history(self, limit=10):

        """Lấy lịch sử chat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_query, bot_response, confidence, timestamp 
            FROM chat_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        history = cursor.fetchall()
        conn.close()
        return history

# Sử dụng Healthcare Knowledge Base
def demo_healthcare_chatbot():
    """Demo Healthcare Chatbot"""
    print("=== HEALTHCARE CHATBOT DEMO ===")
    kb = HealthcareKnowledgeBase()
    
    # Test queries
    test_queries = [
        "Tôi bị ho và sốt, có phải cảm cúm không?",
        "Paracetamol uống như thế nào?",
        "Khi nào tôi cần gọi cấp cứu?",
        "Làm sao để tránh bị tiểu đường?",
        "Tôi bị đau đầu và chóng mặt"
    ]

    while True: 
        print("Câu hỏi: ", end='')
        query = input()
        if query == 'kết thúc':
            print("Cảm ơn bạn đã tin tưởng và sử dụng chatbot")
            break
        response = kb.find_best_answer(query)
        print(f"Bot: {response['answer']}")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Category: {response['category']}")
        print("-" * 50)
        
if __name__ == "__main__":
    demo_healthcare_chatbot()
