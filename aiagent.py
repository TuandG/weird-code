import random

class SimpleAgent:
    def __init__(self):
        self.actions = ["trái", "phải", "lên", "xuống"]

    def choose_action(self, state):
        return random.choice(self.actions)

# Mô phỏng môi trường
state = {"vị trí": (0, 0)}
agent = SimpleAgent()

# Thực hiện 5 hành động
for _ in range(5):
    action = agent.choose_action(state)
    print(f"Hành động được chọn: {action}")
