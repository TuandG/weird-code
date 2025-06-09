from flask import Flask, render_template, request
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CustomClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, num_classes=10):
        super(CustomClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 4, num_classes)

        )


    def forward(self, x):
        return self.network(x)

class HealthcareKnowledgeBase:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """Khởi tạo Knowledge Base cho Healthcare Chatbot"""
        self.model = SentenceTransformer(model_name)
        self.knowledge_data = []
        self.embeddings = []
        self.db_path = "healthcare_kb.db"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_database()
        self.load_healthcare_data()
        self.classifier = CustomClassifier(input_dim=768, hidden_dim=1024, num_classes=len(self.knowledge_data))
        self.classifier.to(self.device)
        if os.path.exists("classifier.pth"):
            self.classifier.load_state_dict(torch.load("classifier.pth"))

            self.classifier.eval()

    def init_database(self):
        """Khởi tạo database SQLite"""
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
        json_file_path = "healthcare_knowledgebase.json"
        if not os.path.exists(json_file_path):
            print(f"Warning: File {json_file_path} không tồn tại!")
            return []
        with open(json_file_path, 'r', encoding='utf-8') as file:
            healthcare_data = json.load(file)
        print(f"Đã load thành công {len(healthcare_data)} câu hỏi từ {json_file_path}")
        

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for item in healthcare_data:
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
        
        self.classifier = CustomClassifier(input_dim=768, hidden_dim=1024, num_classes=len(self.knowledge_data))
        self.classifier.to(self.device)
        
        train_data = [(item['question'], idx) for idx, item in enumerate(self.knowledge_data)]
        print("Number of training samples:", len(train_data))
        print("Train data:", train_data)
        train_data.extend([
            ("Ho và sốt có phải là cảm cúm không?", 0),
            ("Tôi bị sốt và ho, có phải bị cảm không?", 0),
            ("Liều lượng paracetamol là bao nhiêu?", 1),
            ("Làm thế nào để phòng tránh tiểu đường?", 2 if len(self.knowledge_data) > 2 else 0),
        ])

        if train_data:
            self.train_classifier(train_data)
            torch.save(self.classifier.state_dict(), "classifier.pth")

    def train_classifier(self, train_data, epochs=200, lr=0.001, batch_size=32):
        """Huấn luyện classifier trên dữ liệu y tế"""
        optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.classifier.train()
        
        questions = [q for q, _ in train_data]

        labels = [l for _, l in train_data]

        embeddings = self.model.encode(questions)
        dataset = TensorDataset(
            torch.tensor(embeddings, dtype=torch.float32).to(self.device),
            torch.tensor(labels, dtype=torch.long).to(self.device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_embeddings, batch_labels in dataloader:
                optimizer.zero_grad()
                output = self.classifier(batch_embeddings)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
        self.classifier.eval()


    def find_best_answer(self, user_query, threshold=0.5):
        """Tìm câu trả lời phù hợp nhất cho câu hỏi"""
        query_embedding = self.model.encode([user_query])
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).to(self.device)
        

        with torch.no_grad():
            logits = self.classifier(query_tensor)
            probs = torch.softmax(logits, dim=1)
            print("Probabilities:", probs.cpu().numpy())  # Debug
            best_idx = torch.argmax(probs, dim=1).item()
            best_score = probs[0, best_idx].item()
        
        if best_score >= threshold:
            answer = self.knowledge_data[best_idx]['answer']
            category = self.knowledge_data[best_idx]['category']
            self.save_chat_history(user_query, answer, best_score)
            return {"answer": answer, "confidence": float(best_score), "category": category}
        else:
            fallback_answer = "Tôi chưa có thông tin chính xác về vấn đề này. Vui lòng tham khảo ý kiến bác sĩ."
            self.save_chat_history(user_query, fallback_answer, best_score)
            return {"answer": fallback_answer, "confidence": float(best_score), "category": "unknown"}

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

app = Flask(__name__)
kb = HealthcareKnowledgeBase()

@app.route('/', methods=['GET', 'POST'])
def home():
    response = None
    if request.method == 'POST':
        query = request.form.get('query')

        if query:
            response = kb.find_best_answer(query)
            return response
    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
