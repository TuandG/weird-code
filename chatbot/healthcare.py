# Healthcare Chatbot Knowledge Base
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime
import re

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
        """Load dữ liệu healthcare vào knowledge base"""
        healthcare_data = [
            {
                "category": "symptoms",
                "question": "Triệu chứng của cảm cúm là gì?",
                "answer": "Triệu chứng cảm cúm thường bao gồm: sốt, ho, đau họng, nghẹt mũi, đau đầu, đau cơ, mệt mỏi. Nên nghỉ ngơi, uống nhiều nước và tham khảo ý kiến bác sĩ nếu triệu chứng nặng.",
                "keywords": "cảm cúm, sốt, ho, đau họng, nghẹt mũi"
            },
            {
                "category": "symptoms",
                "question": "Dấu hiệu của cao huyết áp?",
                "answer": "Cao huyết áp thường không có triệu chứng rõ ràng, được gọi là 'kẻ giết người thầm lặng'. Một số người có thể gặp: đau đầu, chóng mặt, khó thở, đau ngực. Cần đo huyết áp định kỳ.",
                "keywords": "cao huyết áp, đau đầu, chóng mặt, huyết áp"
            },
            {
                "category": "medication",
                "question": "Cách sử dụng paracetamol an toàn?",
                "answer": "Paracetamol: Liều người lớn 500-1000mg, mỗi 4-6 giờ, tối đa 4g/ngày. Không dùng quá 7 ngày liên tục. Tránh kết hợp với rượu. Thận trọng với người bệnh gan.",
                "keywords": "paracetamol, thuốc giảm đau, liều dùng"
            },
            {
                "category": "emergency",
                "question": "Khi nào cần gọi cấp cứu?",
                "answer": "Gọi cấp cứu 115 khi có: đau ngực dữ dội, khó thở nghiêm trọng, đột quỵ (yếu nửa người, nói khó), chấn thương nặng, mất ý thức, chảy máu không cầm được.",
                "keywords": "cấp cứu, đau ngực, khó thở, đột quỵ, 115"
            },
            {
                "category": "prevention",
                "question": "Cách phòng ngừa tiểu đường type 2?",
                "answer": "Phòng ngừa tiểu đường type 2: Duy trì cân nặng khỏe mạnh, tập thể dục đều đặn, ăn nhiều rau củ, hạn chế đường và carbs tinh chế, không hút thuốc, kiểm tra sức khỏe định kỳ.",
                "keywords": "tiểu đường, phòng ngừa, tập thể dục, ăn uống"
            }
        ]
        
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
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = kb.find_best_answer(query)
        print(f"Bot: {response['answer']}")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Category: {response['category']}")
        print("-" * 50)

if __name__ == "__main__":
    demo_healthcare_chatbot()
