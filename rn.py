import numpy as np

# Môi trường: lưới 3x3, mục tiêu ở (2,2), phần thưởng -1 mỗi bước, +10 tại mục tiêu
Q = np.zeros((9, 4))  # 9 trạng thái, 4 hành động (lên, phải, xuống, trái)
actions = [(0, -3), (0, 1), (0, 3), (0, -1)]  # Thay đổi trạng thái
gamma = 0.9
alpha = 0.1

# Huấn luyện Q-Learning
for _ in range(1000):
    state = 0  # Bắt đầu từ (0,0)
    while state != 8:  # (2,2)
        action = np.random.randint(4)
        next_state = state + actions[action][1]
        if 0 <= next_state < 9:  # Kiểm tra giới hạn
            reward = 10 if next_state == 8 else -1
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state

# Chọn đường đi
state = 0
path = [state]
while state != 8:
    action = np.argmax(Q[state])
    state = state + actions[action][1]
    path.append(state)

print(f"Đường đi tối ưu: {path}")
