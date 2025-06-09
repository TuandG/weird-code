import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import re
from flask import Flask, request, jsonify, render_template_string
import os

# Initialize Flask app

app = Flask(__name__)

# HTML template for the chatbot interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-commerce Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
        #chatbox { width: 100%; height: 400px; border: 1px solid #ddd; overflow-y: scroll; padding: 15px; background-color: #fff; border-radius: 5px; }
        #user_input { width: 75%; padding: 10px; margin-top: 10px; border: 1px solid #ccc; border-radius: 4px; }
        #send_button { padding: 10px 20px; margin-left: 10px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        #send_button:hover { background-color: #0056b3; }
        .message { margin: 10px 0; }
        .user { color: #007bff; font-weight: bold; }
        .bot { color: #28a745; font-weight: bold; }
    </style>
</head>
<body>
    <h1>E-commerce Chatbot</h1>
    <div id="chatbox"></div>
    <input type="text" id="user_input" placeholder="Nhập câu hỏi của bạn...">
    <button id="send_button">Gửi</button>

    <script>
        document.getElementById('send_button').addEventListener('click', sendMessage);
        document.getElementById('user_input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });

        function sendMessage() {

            var userInput = document.getElementById('user_input').value;
            if (userInput.trim() === '') return;
            var chatbox = document.getElementById('chatbox');
            chatbox.innerHTML += '<div class="message"><span class="user">Bạn:</span> ' + userInput + '</div>';
            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: userInput })
            })
            .then(response => response.json())
            .then(data => {
                chatbox.innerHTML += '<div class="message"><span class="bot">Bot:</span> ' + data.answer + '</div>';
                chatbox.scrollTop = chatbox.scrollHeight;
            });
            document.getElementById('user_input').value = '';
        }
    </script>
</body>
</html>
"""

class EcommerceKnowledgeBase:
    def __init__(self, model_path='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_path)
        self.knowledge_data = []
        self.embeddings = []
        self.db_path = "ecommerce_kb.db"
        self.init_database()
        self.load_ecommerce_data()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,

                description TEXT,
                price REAL,
                category TEXT,
                stock INTEGER,
                rating REAL,
                features TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
                customer_email TEXT,
                status TEXT,
                total_amount REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_query TEXT,
                bot_response TEXT,
                intent TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def load_ecommerce_data(self):
        self.load_sample_products()
        try:

            json_file_path = "ecommerce_knowledgebase.json"
            if not os.path.exists(json_file_path):
                print(f"Warning: File {json_file_path} không tồn tại!")
                return
            with open(json_file_path, 'r', encoding='utf-8') as file:
                ecommerce_data = json.load(file)
            print(f"Đã load thành công {len(ecommerce_data)} câu hỏi từ {json_file_path}")
        except Exception as e:
            print(f"Error: {e}")
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for item in ecommerce_data:
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_base 
                (category, question, answer, keywords, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (item['category'], item['question'], item['answer'], item['keywords'], 1.0))
            self.knowledge_data.append(item)
        conn.commit()
        conn.close()
        questions = [item['question'] + " " + item['keywords'] for item in self.knowledge_data]
        self.embeddings = self.model.encode(questions)

    def load_sample_products(self):
        products = [
            {
                "name": "iPhone 15 Pro Max 256GB",
                "description": "iPhone 15 Pro Max với chip A17 Pro, camera 48MP, màn hình 6.7 inch Super Retina XDR",
                "price": 29990000,
                "category": "smartphone",
                "stock": 50,
                "rating": 4.8,
                "features": "chip A17 Pro, camera 48MP, 6.7 inch, titanium"
            },
            {
                "name": "Samsung Galaxy S24 Ultra",
                "description": "Galaxy S24 Ultra với S Pen, camera 200MP, màn hình 6.8 inch Dynamic AMOLED",
                "price": 26990000,
                "category": "smartphone",
                "stock": 30,
                "rating": 4.7,
                "features": "S Pen, camera 200MP, 6.8 inch, AI features"
            },
            {
                "name": "ASUS ROG Strix G15",
                "description": "Laptop gaming ASUS ROG với RTX 4060, AMD Ryzen 7, RAM 16GB",
                "price": 25990000,
                "category": "laptop",
                "stock": 15,
                "rating": 4.6,
                "features": "RTX 4060, Ryzen 7, 16GB RAM, gaming"
            }
        ]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for product in products:
            cursor.execute('''
                INSERT OR REPLACE INTO products 
                (name, description, price, category, stock, rating, features)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (product['name'], product['description'], product['price'],
                  product['category'], product['stock'], product['rating'], product['features']))
        conn.commit()
        conn.close()

    def find_best_answer(self, user_query, threshold=0.5):
        intent = self.classify_intent(user_query)
        if intent == "product_search":
            return self.search_products(user_query)
        if intent == "order_tracking":
            return self.track_order(user_query)

        query_embedding = self.model.encode([user_query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        if best_score >= threshold:
            answer = self.knowledge_data[best_idx]['answer']
            category = self.knowledge_data[best_idx]['category']
            self.save_chat_history(user_query, answer, intent, best_score)
            return {
                "answer": answer,
                "confidence": float(best_score),
                "category": category,
                "intent": intent,
                "source": "knowledge_base"
            }
        else:
            fallback_answer = "Xin lỗi, tôi chưa hiểu rõ câu hỏi của bạn. Bạn có thể liên hệ hotline 1900-xxx-xxx hoặc chat với tư vấn viên để được hỗ trợ tốt hơn."
            self.save_chat_history(user_query, fallback_answer, intent, best_score)
            return {
                "answer": fallback_answer,
                "confidence": float(best_score),
                "category": "unknown",
                "intent": intent,
                "source": "fallback"
            }

    def classify_intent(self, query):
        query_lower = query.lower()
        product_keywords = ['tìm', 'có', 'bán', 'sản phẩm', 'laptop', 'điện thoại', 'iphone', 'samsung', 'giá']
        if any(keyword in query_lower for keyword in product_keywords):
            return "product_search"
        order_keywords = ['đơn hàng', 'order', 'kiểm tra', 'track', 'giao hàng']
        if any(keyword in query_lower for keyword in order_keywords):
            return "order_tracking"
        support_keywords = ['hỗ trợ', 'help', 'liên hệ', 'hotline']
        if any(keyword in query_lower for keyword in support_keywords):
            return "support"
        return "general"

    def search_products(self, query):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query_lower = query.lower()
        if 'iphone' in query_lower or 'điện thoại' in query_lower:
            category_filter = "smartphone"
        elif 'laptop' in query_lower:
            category_filter = "laptop"
        else:
            category_filter = None
        if category_filter:
            cursor.execute('''
                SELECT name, description, price, rating, stock 
                FROM products 
                WHERE category = ? AND stock > 0
                ORDER BY rating DESC
            ''', (category_filter,))
        else:
            cursor.execute('''
                SELECT name, description, price, rating, stock 
                FROM products 
                WHERE stock > 0
                ORDER BY rating DESC
            ''')
        products = cursor.fetchall()
        conn.close()
        if products:
            response = "Tôi tìm thấy các sản phẩm phù hợp:\n\n"
            for product in products[:3]:
                name, desc, price, rating, stock = product
                response += f"🔸 {name}\n"
                response += f"   Giá: {price:,.0f}đ\n"
                response += f"   Đánh giá: {rating}⭐ | Còn: {stock} sản phẩm\n"
                response += f"   {desc[:100]}...\n\n"
            response += "Bạn có muốn xem chi tiết sản phẩm nào không?"
            return {
                "answer": response,
                "confidence": 0.9,
                "category": "product_search",
                "intent": "product_search",
                "source": "product_database",
                "products": products
            }
        else:
            return {
                "answer": "Xin lỗi, hiện tại chúng tôi chưa có sản phẩm phù hợp với yêu cầu của bạn. Vui lòng thử tìm kiếm với từ khóa khác.",

                "confidence": 0.7,
                "category": "product_search",
                "intent": "product_search",
                "source": "no_results"
            }

    def track_order(self, query):
        order_pattern = r'[A-Z]{2}\d{6}'
        order_match = re.search(order_pattern, query.upper())
        if order_match:
            order_id = order_match.group()
            mock_orders = {
                "DH123456": {"status": "Đang giao hàng", "estimated": "2 ngày nữa"},
                "DH789012": {"status": "Đã giao", "delivered": "hôm qua"},
                "DH345678": {"status": "Đang xử lý", "estimated": "1-2 ngày nữa"}
            }
            if order_id in mock_orders:
                order_info = mock_orders[order_id]
                response = f"Thông tin đơn hàng {order_id}:\n"
                response += f"📦 Trạng thái: {order_info['status']}\n"
                if 'estimated' in order_info:
                    response += f"⏰ Dự kiến: {order_info['estimated']}\n"
                if 'delivered' in order_info:
                    response += f"✅ Đã giao: {order_info['delivered']}\n"
                response += "\nCảm ơn bạn đã mua hàng!"
                return {
                    "answer": response,
                    "confidence": 0.95,
                    "category": "order_tracking",
                    "intent": "order_tracking",
                    "source": "order_database"
                }
        return {
            "answer": "Để tra cứu đơn hàng, vui lòng cung cấp mã đơn hàng (VD: DH123456) hoặc số điện thoại đặt hàng.",
            "confidence": 0.8,

            "category": "order_tracking",
            "intent": "order_tracking",
            "source": "missing_info"
        }

    def save_chat_history(self, query, response, intent, confidence):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_history (user_query, bot_response, intent, confidence)
            VALUES (?, ?, ?, ?)
        ''', (query, response, intent, confidence))
        conn.commit()
        conn.close()

# Initialize the knowledge base
model_path = 'fine_tuned_ecommerce_model' if os.path.exists('fine_tuned_ecommerce_model') else 'all-MiniLM-L6-v2'
kb = EcommerceKnowledgeBase(model_path=model_path)

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    response = kb.find_best_answer(user_query)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
