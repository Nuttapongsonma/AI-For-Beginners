import torch
import torch.nn as nn

# สร้างคลาส MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# สร้างโมเดลและกำหนด optimizer
model = MLP()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# แปลงข้อมูลเป็น tensor
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.FloatTensor([[0], [1], [1], [0]])

# ฝึกฝนโมเดล
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# ทำนายผล
with torch.no_grad():
    predictions = model(X)
    print("Predictions:", predictions.numpy())
