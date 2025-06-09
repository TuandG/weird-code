from torch import optim, nn
def train_classifier(self, train_data, epochs=10, lr=0.001):
    """Huấn luyện classifier trên dữ liệu y tế"""
    optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    self.classifier.train()
    
    for epoch in range(epochs):

        total_loss = 0
        for question, label in train_data:
            embedding = self.model.encode([question])
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)

            
            optimizer.zero_grad()
            output = self.classifier(embedding_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data)}")
    self.classifier.eval()
