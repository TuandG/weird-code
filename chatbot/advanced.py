# Advanced Knowledge Base Features
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any, Tuple
import logging
from collections import Counter, defaultdict

class AdvancedKnowledgeBase:

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Advanced Knowledge Base với các tính năng nâng cao:

        - Multi-turn conversation
        - Context awareness
        - Learning from interactions
        - Analytics và insights
        - A/B testing responses
        """
        self.model = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.knowledge_data = []
        self.embeddings = []
        self.conversation_context = {}
        self.db_path = "advanced_kb.db"
        self.setup_logging()
        self.init_advanced_database()
        self.load_advanced_features()
    
    def setup_logging(self):
        """Setup logging cho monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chatbot.log'),

                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def init_advanced_database(self):
        """Khởi tạo database nâng cao"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng knowledge base với versioning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER DEFAULT 1,
                category TEXT,
                question TEXT,
                answer TEXT,
                keywords TEXT,
                confidence_score REAL,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng conversation context
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                context_data TEXT,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng feedback và learning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT,
                response TEXT,
                rating INTEGER,
                feedback_text TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng A/B testing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_testing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                variant TEXT,

                query TEXT,
                response TEXT,
                user_reaction TEXT,
                conversion_rate REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP

            )
        ''')
        
        # Bảng analytics events

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id TEXT,

                event_type TEXT,

                event_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng intent classification training
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intent_training (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                intent TEXT,
                confidence REAL,
                is_correct BOOLEAN,
                feedback_source TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_advanced_features(self):
        """Load các tính năng nâng cao"""
        # Load knowledge base với performance metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''

            SELECT category, question, answer, keywords, usage_count, success_rate
            FROM knowledge_base 
            WHERE version = (SELECT MAX(version) FROM knowledge_base)
            ORDER BY success_rate DESC, usage_count DESC
        ''')

        
        kb_data = cursor.fetchall()
        conn.close()
        
        if not kb_data:
            # Load default data nếu chưa có
            self.load_default_knowledge()
        else:
            for row in kb_data:
                self.knowledge_data.append({
                    'category': row[0],
                    'question': row[1], 
                    'answer': row[2],
                    'keywords': row[3],
                    'usage_count': row[4],
                    'success_rate': row[5]
                })
        
        # Tạo embeddings
        if self.knowledge_data:
            questions = [item['question'] + " " + item['keywords'] for item in self.knowledge_data]
            self.embeddings = self.model.encode(questions)
    
    def load_default_knowledge(self):
        """Load knowledge mặc định"""
        default_knowledge = [
            {
                "category": "greeting",
                "question": "xin chào hello hi",
                "answer": "Xin chào! Tôi là trợ lý AI của bạn. Tôi có thể giúp bạn tìm hiểu thông tin, trả lời câu hỏi và hỗ trợ các vấn đề khác nhau. Bạn cần hỗ trợ gì hôm nay?",
                "keywords": "chào hỏi greeting hello hi"
            },
            {
                "category": "help",
                "question": "tôi cần giúp đỡ help support",
                "answer": "Tôi có thể hỗ trợ bạn trong nhiều lĩnh vực: trả lời câu hỏi, tìm kiếm thông tin, giải thích khái niệm, tư vấn và nhiều việc khác. Hãy cho tôi biết cụ thể bạn cần hỗ trợ gì nhé!",
                "keywords": "giúp đỡ help support hỗ trợ"
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in default_knowledge:
            cursor.execute('''
                INSERT INTO knowledge_base 
                (category, question, answer, keywords, confidence_score, usage_count, success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (item['category'], item['question'], item['answer'], 
                  item['keywords'], 1.0, 0, 0.0))
            
            self.knowledge_data.append({
                **item,
                'usage_count': 0,
                'success_rate': 0.0
            })
        
        conn.commit()
        conn.close()
    
    def process_query_with_context(self, user_query: str, session_id: str, user_id: str = None) -> Dict[str, Any]:
        """Xử lý query với context awareness"""
        
        # Log sự kiện
        self.log_analytics_event(session_id, "query_received", {"query": user_query})
        
        # Lấy context của conversation
        context = self.get_conversation_context(session_id)
        
        # Phân tích intent với context
        intent_info = self.advanced_intent_classification(user_query, context)
        
        # Tìm câu trả lời với context
        response_info = self.find_contextual_answer(user_query, context, intent_info)
        
        # Cập nhật context
        self.update_conversation_context(session_id, user_id, user_query, response_info, intent_info)
        
        # Log response
        self.log_analytics_event(session_id, "response_sent", {
            "intent": intent_info['intent'],
            "confidence": response_info['confidence'],
            "source": response_info['source']
        })
        
        return response_info
    
    def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Lấy context của cuộc trò chuyện"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT context_data, last_activity 
            FROM conversation_context 
            WHERE session_id = ?
            ORDER BY last_activity DESC 
            LIMIT 1
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                context_data = json.loads(result[0])
                # Kiểm tra context có hết hạn không (30 phút)
                last_activity = datetime.fromisoformat(result[1])

                if datetime.now() - last_activity > timedelta(minutes=30):
                    return {"conversation_history": [], "current_topic": None, "user_preferences": {}}
                return context_data
            except:
                pass
        
        return {"conversation_history": [], "current_topic": None, "user_preferences": {}}
    
    def advanced_intent_classification(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phân loại intent nâng cao với context"""
        query_lower = query.lower()
        
        # Phân tích context để hiểu intent tốt hơn
        current_topic = context.get('current_topic')
        conversation_history = context.get('conversation_history', [])
        
        # Intent patterns nâng cao

        intent_patterns = {
            'greeting': ['xin chào', 'hello', 'hi', 'chào bạn'],
            'question': ['là gì', 'như thế nào', 'tại sao', 'làm sao', '?'],
            'request': ['tôi muốn', 'bạn có thể', 'giúp tôi', 'làm giúp'],
            'complaint': ['không hài lòng', 'có vấn đề', 'lỗi', 'không hoạt động'],
            'compliment': ['tốt', 'hay', 'cảm ơn', 'xuất sắc'],
            'goodbye': ['tạm biệt', 'bye', 'kết thúc', 'xong rồi']
        }
        
        # Tính điểm cho mỗi intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            
            # Boost score dựa trên context
            if current_topic and intent in ['question', 'request']:
                score += 0.5
            

            intent_scores[intent] = score
        

        # Lấy intent có điểm cao nhất
        best_intent = max(intent_scores.items(), key=lambda x: x[1])

        
        return {
            'intent': best_intent[0] if best_intent[1] > 0 else 'general',

            'confidence': min(best_intent[1] / 3.0, 1.0),

            'all_scores': intent_scores,
            'context_influenced': current_topic is not None

        }
    
    def find_contextual_answer(self, query: str, context: Dict[str, Any], intent_info: Dict[str, Any]) -> Dict[str, Any]:

        """Tìm câu trả lời với context awareness"""
        

        # Thêm context vào query để tìm kiếm tốt hơn
        enhanced_query = query
        if context.get('current_topic'):
            enhanced_query = f"{context['current_topic']} {query}"
        
        # Tìm trong knowledge base
        if self.embeddings is not None and len(self.embeddings) > 0:
            query_embedding = self.model.encode([enhanced_query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            threshold = 0.4  # Giảm threshold để linh hoạt hơn
            
            if best_score >= threshold:
                answer_data = self.knowledge_data[best_idx]
                
                # Personalize answer dựa trên context
                answer = self.personalize_answer(answer_data['answer'], context, intent_info)
                
                # Cập nhật usage statistics
                self.update_knowledge_usage(best_idx, True)
                

                return {
                    "answer": answer,
                    "confidence": float(best_score),
                    "category": answer_data['category'],
                    "intent": intent_info['intent'],
                    "source": "knowledge_base",
                    "context_used": bool(context.get('current_topic')),
                    "personalized": True
                }
        
        # Fallback với context-aware response
        fallback_answer = self.generate_contextual_fallback(query, context, intent_info)
        
        return {
            "answer": fallback_answer,
            "confidence": 0.3,
            "category": "unknown",
            "intent": intent_info['intent'],
            "source": "fallback",
            "context_used": bool(context.get('current_topic')),
            "personalized": False
        }
    
    def personalize_answer(self, base_answer: str, context: Dict[str, Any], intent_info: Dict[str, Any]) -> str:
        """Cá nhân hóa câu trả lời"""
        answer = base_answer
        
        # Thêm context reference nếu có
        if context.get('current_topic'):
            if not any(topic in answer.lower() for topic in [context['current_topic'].lower()]):
                answer = f"Liên quan đến {context['current_topic']}: {answer}"
        
        # Thêm follow-up questions dựa trên intent
        if intent_info['intent'] == 'question':
            answer += "\n\nBạn có muốn biết thêm thông tin gì khác không?"
        elif intent_info['intent'] == 'request':
            answer += "\n\nTôi có thể hỗ trợ thêm gì khác cho bạn?"
        
        return answer
    
    def generate_contextual_fallback(self, query: str, context: Dict[str, Any], intent_info: Dict[str, Any]) -> str:
        """Tạo fallback response có context"""
        base_fallback = "Xin lỗi, tôi chưa hiểu rõ câu hỏi của bạn."
        
        # Customize dựa trên intent
        if intent_info['intent'] == 'greeting':
            return "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"
        elif intent_info['intent'] == 'goodbye':
            return "Tạm biệt! Chúc bạn một ngày tốt lành!"
        elif intent_info['intent'] == 'compliment':
            return "Cảm ơn bạn! Tôi rất vui khi được hỗ trợ bạn."

        
        # Thêm context hint
        if context.get('current_topic'):
            base_fallback += f" Chúng ta đang nói về {context['current_topic']}, bạn có thể diễn đạt lại câu hỏi được không?"
        
        base_fallback += " Bạn có thể thử hỏi theo cách khác hoặc cung cấp thêm chi tiết?"
        
        return base_fallback
    
    def update_conversation_context(self, session_id: str, user_id: str, query: str, response_info: Dict[str, Any], intent_info: Dict[str, Any]):
        """Cập nhật context cuộc trò chuyện"""
        current_context = self.get_conversation_context(session_id)
        

        # Cập nhật conversation history
        current_context['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intent': intent_info['intent'],
            'category': response_info.get('category'),
            'confidence': response_info.get('confidence')
        })
        
        # Chỉ giữ 10 turn gần nhất
        if len(current_context['conversation_history']) > 10:
            current_context['conversation_history'] = current_context['conversation_history'][-10:]
        
        # Cập nhật current topic
        if response_info.get('category') and response_info.get('category') != 'unknown':
            current_context['current_topic'] = response_info['category']
        
        # Lưu context
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO conversation_context 
            (session_id, user_id, context_data, last_activity)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_id, json.dumps(current_context), datetime.now().isoformat()))
        conn.commit()
        conn.close()
    

    def update_knowledge_usage(self, knowledge_idx: int, was_helpful: bool):
        """Cập nhật thống kê sử dụng knowledge"""
        if knowledge_idx >= len(self.knowledge_data):
            return
            
        # Cập nhật in-memory data
        self.knowledge_data[knowledge_idx]['usage_count'] += 1

        
        # Tính success rate
        current_success = self.knowledge_data[knowledge_idx].get('success_rate', 0.0)
        current_usage = self.knowledge_data[knowledge_idx]['usage_count']
        
        if was_helpful:

            new_success_rate = ((current_success * (current_usage - 1)) + 1.0) / current_usage
        else:
            new_success_rate = (current_success * (current_usage - 1)) / current_usage
        
        self.knowledge_data[knowledge_idx]['success_rate'] = new_success_rate
        
        # Cập nhật database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE knowledge_base 
            SET usage_count = usage_count + 1,
                success_rate = ?,
                last_updated = ?
            WHERE question = ? AND answer = ?
        ''', (new_success_rate, datetime.now().isoformat(),
              self.knowledge_data[knowledge_idx]['question'],
              self.knowledge_data[knowledge_idx]['answer']))
        conn.commit()
        conn.close()
    
    def log_analytics_event(self, session_id: str, event_type: str, event_data: Dict[str, Any]):
        """Log analytics events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analytics_events (session_id, event_type, event_data)
            VALUES (?, ?, ?)
        ''', (session_id, event_type, json.dumps(event_data)))
        conn.commit()

        conn.close()
        
        self.logger.info(f"Analytics: {event_type} - {session_id} - {event_data}")

    
    def collect_user_feedback(self, session_id: str, query: str, response: str, rating: int, feedback_text: str = ""):
        """Thu thập feedback từ user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_feedback (session_id, query, response, rating, feedback_text)
            VALUES (?, ?, ?, ?, ?)

        ''', (session_id, query, response, rating, feedback_text))
        conn.commit()
        conn.close()
        

        # Học từ feedback
        self.learn_from_feedback(query, response, rating, feedback_text)
    
    def learn_from_feedback(self, query: str, response: str, rating: int, feedback_text: str):
        """Học từ user feedback"""
        # Nếu rating thấp (<3), mark response là không helpful
        if rating < 3:
            # Tìm knowledge item tương ứng và giảm success rate
            query_embedding = self.model.encode([query])
            if len(self.embeddings) > 0:
                similarities = cosine_similarity(query_embedding, self.embeddings)[0]
                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.5:  # Nếu match
                    self.update_knowledge_usage(best_idx, False)

        
        # Log để phân tích sau
        self.logger.info(f"Feedback received: Rating={rating}, Query='{query[:50]}...', Feedback='{feedback_text[:100]}...'")

    
    def get_advanced_analytics(self) -> Dict[str, Any]:
        """Lấy analytics nâng cao"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Intent distribution
        cursor.execute('''
            SELECT JSON_EXTRACT(event_data, '$.intent') as intent, COUNT(*) as count
            FROM analytics_events 
            WHERE event_type = 'response_sent' AND JSON_EXTRACT(event_data, '$.intent') IS NOT NULL
            GROUP BY intent
            ORDER BY count DESC
        ''')
        intent_dist = cursor.fetchall()
        
        # Confidence distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN CAST(JSON_EXTRACT(event_data, '$.confidence') AS REAL) >= 0.8 THEN 'High'

                    WHEN CAST(JSON_EXTRACT(event_data, '$.confidence') AS REAL) >= 0.5 THEN 'Medium'
                    ELSE 'Low'
                END as confidence_level,
                COUNT(*) as count
            FROM analytics_events 

            WHERE event_type = 'response_sent'
            GROUP BY confidence_level

        ''')

        confidence_dist = cursor.fetchall()
        
        # User feedback stats
        cursor.execute('''
            SELECT AVG(rating) as avg_rating, COUNT(*) as total_feedback
            FROM user_feedback
        ''')
        feedback_stats = cursor.fetchone()
        
        # Top performing knowledge
        cursor.execute('''
            SELECT question, usage_count, success_rate, category
            FROM knowledge_base
            ORDER BY (usage_count * success_rate) DESC
            LIMIT 10
        ''')
        top_knowledge = cursor.fetchall()
        
        conn.close()
        
        return {
            "intent_distribution": dict(intent_dist) if intent_dist else {},
            "confidence_distribution": dict(confidence_dist) if confidence_dist else {},
            "average_rating": feedback_stats[0] if feedback_stats[0] else 0.0,
            "total_feedback": feedback_stats[1] if feedback_stats[1] else 0,
            "top_performing_knowledge": [
                {
                    "question": item[0][:100] + "..." if len(item[0]) > 100 else item[0],
                    "usage_count": item[1],
                    "success_rate": round(item[2], 2),
                    "category": item[3]
                } for item in top_knowledge
            ] if top_knowledge else []
        }
    
    def optimize_knowledge_base(self):
        """Tự động tối ưu knowledge base"""
        # Xóa knowledge có success rate thấp và ít sử dụng
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tìm knowledge cần cải thiện
        cursor.execute('''
            SELECT id, question, answer, usage_count, success_rate
            FROM knowledge_base
            WHERE usage_count > 10 AND success_rate < 0.3
        ''')
        
        low_performing = cursor.fetchall()
        
        if low_performing:
            self.logger.info(f"Found {len(low_performing)} low-performing knowledge items for review")
            
            # Mark for review instead of auto-delete
            for item in low_performing:
                cursor.execute('''
                    UPDATE knowledge_base 
                    SET keywords = keywords || ' [NEEDS_REVIEW]'
                    WHERE id = ?
                ''', (item[0],))
        
        # Cluster similar questions để tìm duplicate

        if len(self.knowledge_data) > 5:

            questions = [item['question'] for item in self.knowledge_data]
            question_embeddings = self.model.encode(questions)
            
            # Simple clustering
            n_clusters = min(len(questions) // 3, 10)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(question_embeddings)
                
                # Group by clusters to find potential duplicates
                cluster_groups = defaultdict(list)
                for i, cluster_id in enumerate(clusters):
                    cluster_groups[cluster_id].append((i, questions[i]))
                
                duplicates_found = 0
                for cluster_id, items in cluster_groups.items():
                    if len(items) > 1:
                        # Check similarity within cluster
                        for i in range(len(items)):
                            for j in range(i+1, len(items)):
                                idx1, q1 = items[i]
                                idx2, q2 = items[j]
                                similarity = cosine_similarity(
                                    [question_embeddings[idx1]], 
                                    [question_embeddings[idx2]]
                                )[0][0]
                                
                                if similarity > 0.9:  # Very similar
                                    duplicates_found += 1
                                    self.logger.info(f"Potential duplicate found: '{q1[:50]}...' and '{q2[:50]}...'")
                
                if duplicates_found > 0:
                    self.logger.info(f"Found {duplicates_found} potential duplicate questions")
        

        conn.commit()
        conn.close()

# Demo Advanced Knowledge Base
def demo_advanced_kb():
    """Demo Advanced Knowledge Base"""
    print("=== ADVANCED KNOWLEDGE BASE DEMO ===")
    
    kb = AdvancedKnowledgeBase()
    
    # Simulate conversation with context
    session_id = "demo_session_001"
    user_id = "demo_user"
    
    queries = [
        "Xin chào!",
        "Tôi muốn hỏi về sản phẩm",
        "Có laptop gaming không?",
        "Giá bao nhiêu?",
        "Còn có màu nào khác?",
        "Cảm ơn bạn nhiều!"
    ]
    
    print("\n--- Conversation Flow ---")
    for i, query in enumerate(queries):

        print(f"\nTurn {i+1}")
        print(f"User: {query}")
        
        response = kb.process_query_with_context(query, session_id, user_id)
        print(f"Bot: {response['answer']}")
        print(f"Intent: {response['intent']} | Confidence: {response['confidence']:.2f}")
        print(f"Context Used: {response.get('context_used', False)}")
        
        # Simulate user feedback occasionally
        if i % 2 == 0:
            rating = np.random.choice([4, 5], p=[0.3, 0.7])  # Mostly positive
            kb.collect_user_feedback(session_id, query, response['answer'], rating, "Helpful response")

            print(f"[Feedback: {rating}/5]")
        
        print("-" * 50)
    
    # Show analytics
    print("\n=== ADVANCED ANALYTICS ===")
    analytics = kb.get_advanced_analytics()
    
    print(f"Average Rating: {analytics['average_rating']:.2f}")

    print(f"Total Feedback: {analytics['total_feedback']}")

    print(f"Intent Distribution: {analytics['intent_distribution']}")
    print(f"Confidence Distribution: {analytics['confidence_distribution']}")
    
    print("\nTop Performing Knowledge:")
    for item in analytics['top_performing_knowledge'][:3]:
        print(f"- {item['question']} (Used: {item['usage_count']}, Success: {item['success_rate']})")
    

    # Demonstrate optimization
    print("\n=== KNOWLEDGE BASE OPTIMIZATION ===")
    kb.optimize_knowledge_base()
    print("Optimization completed. Check logs for details.")

if __name__ == "__main__":
    demo_advanced_kb()
