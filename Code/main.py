# -------------------------------------------------------------------
# Assembler-Ms-Ds: первая обучаемая версия нейросети beta 1.0
# Дата выхода - 18.12.2025
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import difflib

# -------------------------------------------------------------------
# 1. Подготовка данных
# -------------------------------------------------------------------

non_pc_text = list(line.strip() for line in open('text_not_pc.txt', encoding="utf-8").readlines())
game_pc_text = list(line.strip() for line in open('game_pc_with.txt', encoding="utf-8").readlines())
office_pc_text = list(line.strip() for line in open('office_pc_with.txt', encoding="utf-8").readlines())

cpus = [
    {"name": "Ryzen 5 5600", "score": 0.75, "socket": "AM4", "price": 12000},
    {"name": "i5-12400F", "score": 0.78, "socket": "LGA1700", "price": 14000},
]

gpus = [
    {"name": "RTX 3060", "score": 0.8, "power": 170, "price": 25000},
    {"name": "RTX 4060", "score": 0.9, "power": 160, "price": 30000},
]

rams = [
    {"size": 16, "price": 4000},
    {"size": 32, "price": 7000},
]

ssds = [
    {"size": 512, "price": 3000},
    {"size": 1000, "price": 5000},
]

psus = [
    {"power": 600, "price": 5000},
    {"power": 750, "price": 7000},
]

# -------------------------------------------------------------------
# 2. Функции подготовки данных
# -------------------------------------------------------------------

def is_compatible(cpu, gpu, psu): # Проверка совместимости CPU, GPU и блока питания
    return psu['power'] >= gpu['power'] + 200

def find_similar(word, vocabulary, cutoff=0.6): # Поиск похожего слова в словаре для защиты от опечаток
    matches = difflib.get_close_matches(word, vocabulary, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def text_to_vec(text, word2ixd, vocab_size): # Превращает текст в вектор признаков для нейросети
    vec = torch.zeros(vocab_size)
    for w in text.lower():
        if w in word2ixd:
            vec[word2ixd[w]] = 1
        else:
            similar = find_similar(w, word2ixd.keys())
            if similar:
                vec[word2ixd[similar]] = 1

    return vec

def build_to_tensor(build): # Превращает сборку ПК в тензор для оценщика
    return torch.tensor([[
        build['cpu']['score'],
        build['gpu']['score'],
        build['ram']['size'],
        build['ssd']['size'],
        build['psu']['power'],
        build['price'],
    ]], dtype=torch.float32)

# -------------------------------------------------------------------
# 3. Обучение текстовой нейросети
# -------------------------------------------------------------------

text = non_pc_text + game_pc_text + office_pc_text # Создание словаря58
labels = [0]*len(non_pc_text) + [1]*len(game_pc_text) + [2]*len(office_pc_text) # таблица коэффициентов
words = set(w for t in text for w in t.split())
word2ixd = {w: i for i, w in enumerate(words)}
vocab_size = len(words)

class PCClassifier(nn.Module): # Класс классификатора текста
    def __init__(self, input_size):
        super().__init__()
        self.layer = nn.Linear(input_size, 3)

    def forward(self, x):
        return self.layer(x)

data = torch.stack([text_to_vec(t, word2ixd, vocab_size) for t in text]) # Данные
targets = torch.LongTensor(labels)

text_model = PCClassifier(vocab_size) # Обучение модели
loss_fn = nn.CrossEntropyLoss()
optim_model = optim.SGD(text_model.parameters(), lr=0.1)

print("Начинаем обучение текстовой нейросети...")

for epoch in range(3001):
    logits = text_model(data)
    loss = loss_fn(logits, targets)
    optim_model.zero_grad()
    loss.backward()
    optim_model.step()

    if epoch % 1000 == 0:
        print(f'{(epoch / 3000):.2f}')

print("Обучение завершено!")

# -------------------------------------------------------------------с
# 4. Нейросеть - оценщик сборки
# -------------------------------------------------------------------

class PCScoringNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

scoring_model = PCScoringNet()
scoring_model.eval() # пока без обучения

# -------------------------------------------------------------------
# 5. Генератор сборки
# -------------------------------------------------------------------

def generate_build(pc_type, budget):
    if pc_type == 1: # игровой
        allowed_gpus = gpus
    else: # офисный
        allowed_gpus = [{"name": "iGPU", "score": 0.2, "power": 0, "price": 0}]

    builds = []
    for cpu  in cpus:
        for gpu in allowed_gpus:
            for ram in rams:
                for ssd in ssds:
                    for psu in psus:
                        price = cpu["price"] + gpu["price"] + ram["price"] + ssd["price"] + psu["price"]
                        if price > budget:
                            continue
                        if not is_compatible(cpu, gpu, psu):
                            continue
                        builds.append({"cpu": cpu, "gpu": gpu, "ram": ram, "ssd": ssd, "psu": psu, "price": price})

    return builds

def choose_best_build(build, model):
    best_score = -1
    best_build = None

    for build in builds:
        x = build_to_tensor(build)
        score = model(x).item()
        if score > best_score:
            best_score = score
            best_build = build

    return best_build, best_score

# -------------------------------------------------------------------
# 6. Анализ текста
# -------------------------------------------------------------------

def analyze_text(text):
    vec = text_to_vec(text, word2ixd, vocab_size)
    with torch.no_grad():
        logits = text_model(vec)
        cls = torch.argmax(logits).item()

    if cls == 0:
        return 0, 'Вы задали вопрос не по теме или я не распознал запрос. Попробуйте задать его снова'
    elif cls == 1:
        return 1, 'Вижу, что вы хотите игровой пк. Могу предложить вам компьютер со следующими характеристиками:'
    else:
        return 2, 'Вижу, что вы хотите офисный пк. Могу предложить вам компьютер со следующими характеристиками:'

# -------------------------------------------------------------------
# 7. Основной цикл
# -------------------------------------------------------------------

print('Остановка нейросети — командой "стоп"')

while True:
    req = input("Введите запрос: ").lower()

    if req == 'стоп':
        print("Завершение работы...")
        break

    cls, message = analyze_text(req)
    print(message)

    if cls in (1, 2):
        builds = generate_build(cls, budget=80000)

        if not builds:
            print("Не удалось подобрать сборку под бюджет")
            continue

        best_build, score = choose_best_build(builds, scoring_model)

        print("\nРекомендуемая сборка:")
        print("CPU:", best_build["cpu"]["name"])
        print("GPU:", best_build["gpu"]["name"])
        print("RAM:", best_build["ram"]["size"], "ГБ")
        print("SSD:", best_build["ssd"]["size"], "ГБ")
        print("БП:", best_build["psu"]["power"], "Вт")
        print("Цена:", best_build["price"])
        print("Оценка нейросети:", 'N/A')
        print('')