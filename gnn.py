import torch

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# Tạo dữ liệu đồ thị đơn giản
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Cạnh
x = torch.tensor([[1], [2], [3]], dtype=torch.float)  # Đặc trưng của nút
y = torch.tensor([0, 1, 0], dtype=torch.long)  # Nhãn của nút

data = Data(x=x, edge_index=edge_index, y=y)

# Xây dựng mô hình GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(1, 2)
        self.conv2 = GCNConv(2, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Khởi tạo và huấn luyện
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for _ in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()


# Dự đoán
model.eval()
pred = out.argmax(dim=1)
print(f"Dự đoán nhãn của các nút: {pred.tolist()}")
