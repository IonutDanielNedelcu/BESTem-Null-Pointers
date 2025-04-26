# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# # 1. Setup
# checkpoint_path = "D:/University/BEST/BEST/bert_fight_classifier/bert_fight_classifier"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
#
# # Trimitem modelul pe GPU dacă este disponibil
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# model.eval()
#
#
# # 2. Construim mapele
# words_list = [
#     "sandpaper", "oil", "steam", "acid", "gust", "boulder", "drill", "vacation", "fire", "drought", "water",
#     "vacuum", "laser", "life raft", "bear trap", "hydraulic jack", "diamond cage", "dam", "sunshine",
#     "mutation", "kevlar vest", "jackhammer", "signal jammer", "grizzly", "reinforced steel door",
#     "bulldozer", "sonic boom", "robot", "glacier", "love", "fire blanket", "super glue", "therapy", "disease",
#     "fire extinguisher", "satellite", "confidence", "absorption", "neutralizing agent", "freeze", "encryption",
#     "proof", "molotov cocktail", "rainstorm", "viral meme", "war", "dynamite", "seismic dampener",
#     "propaganda", "explosion", "lightning", "evacuation", "flood", "lava", "reforestation", "avalanche",
#     "earthquake", "h-bomb", "dragon", "innovation", "hurricane", "tsunami", "persistence", "resilience",
#     "terraforming device", "anti-virus nanocloud", "ai kill switch", "nanobot swarm", "reality resynchronizer",
#     "cataclysm containment field", "solar deflection array", "planetary evacuation fleet", "antimatter cannon",
#     "planetary defense shield", "singularity stabilizer", "orbital laser", "time"
# ]
#
# words_dict = {
#     "sandpaper": [8, 1],
#     "oil": [10, 2],
#     "steam": [15, 3],
#     "acid": [16, 4],
#     "gust": [18, 5],
#     "boulder": [20, 6],
#     "drill": [20, 7],
#     "vacation": [20, 8],
#     "fire": [22, 9],
#     "drought": [24, 10],
#     "water": [25, 11],
#     "vacuum": [27, 12],
#     "laser": [28, 13],
#     "life raft": [30, 14],
#     "bear trap": [32, 15],
#     "hydraulic jack": [33, 16],
#     "diamond cage": [35, 17],
#     "dam": [35, 18],
#     "sunshine": [35, 19],
#     "mutation": [35, 20],
#     "kevlar vest": [38, 21],
#     "jackhammer": [38, 22],
#     "signal jammer": [40, 23],
#     "grizzly": [41, 24],
#     "reinforced steel door": [42, 25],
#     "bulldozer": [42, 26],
#     "sonic boom": [45, 27],
#     "robot": [45, 28],
#     "glacier": [45, 29],
#     "love": [45, 30],
#     "fire blanket": [48, 31],
#     "super glue": [48, 32],
#     "therapy": [48, 33],
#     "disease": [50, 34],
#     "fire extinguisher": [50, 35],
#     "satellite": [50, 36],
#     "confidence": [50, 37],
#     "absorption": [52, 38],
#     "neutralizing agent": [55, 39],
#     "freeze": [55, 40],
#     "encryption": [55, 41],
#     "proof": [55, 42],
#     "molotov cocktail": [58, 43],
#     "rainstorm": [58, 44],
#     "viral meme": [58, 45],
#     "war": [59, 46],
#     "dynamite": [60, 47],
#     "seismic dampener": [60, 48],
#     "propaganda": [60, 49],
#     "explosion": [62, 50],
#     "lightning": [65, 51],
#     "evacuation": [65, 52],
#     "flood": [67, 53],
#     "lava": [68, 54],
#     "reforestation": [70, 55],
#     "avalanche": [72, 56],
#     "earthquake": [74, 57],
#     "h-bomb": [75, 58],
#     "dragon": [75, 59],
#     "innovation": [75, 60],
#     "hurricane": [76, 61],
#     "tsunami": [78, 62],
#     "persistence": [80, 63],
#     "resilience": [85, 64],
#     "terraforming device": [89, 65],
#     "anti-virus nanocloud": [90, 66],
#     "ai kill switch": [90, 67],
#     "nanobot swarm": [92, 68],
#     "reality resynchronizer": [92, 69],
#     "cataclysm containment field": [92, 70],
#     "solar deflection array": [93, 71],
#     "planetary evacuation fleet": [94, 72],
#     "antimatter cannon": [95, 73],
#     "planetary defense shield": [96, 74],
#     "singularity stabilizer": [97, 75],
#     "orbital laser": [98, 76],
#     "time": [100, 77]
# }
#
# word2id = {word.lower(): idx for idx, word in enumerate(words_list)}
# id2word = {idx: word for word, idx in word2id.items()}
#
#
# # 3. Funcție ultra-rapidă
# @torch.no_grad()
# def predict(cuvant: str) -> int:
#     inputs = tokenizer(cuvant, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}  # Mutăm și inputurile pe device
#
#     logits = model(**inputs).logits.squeeze()
#
#     probs = torch.softmax(logits, dim=-1)
#     top3_probs, top3_indices = probs.topk(3)
#
#     best_score = 101
#     best_id = 0
#
#     for idx in top3_indices:
#         idx = idx.item()
#         word = id2word.get(idx, "necunoscut")
#         score, word_id = words_dict.get(word, (999, -1))
#         if score < best_score:
#             best_score = score
#             best_id = word_id
#
#     return best_id
#
# # 4. Exemplu de folosire
# print(predict("water"))



import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

start_time = time.time()  # Pornim cronometrul

# 1. Setup
checkpoint_path = "D:/University/BEST/BEST/bert_fight_classifier/bert_fight_classifier"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


# 2. Construim mapele
words_list = [
    "sandpaper", "oil", "steam", "acid", "gust", "boulder", "drill", "vacation", "fire", "drought", "water",
    "vacuum", "laser", "life raft", "bear trap", "hydraulic jack", "diamond cage", "dam", "sunshine",
    "mutation", "kevlar vest", "jackhammer", "signal jammer", "grizzly", "reinforced steel door",
    "bulldozer", "sonic boom", "robot", "glacier", "love", "fire blanket", "super glue", "therapy", "disease",
    "fire extinguisher", "satellite", "confidence", "absorption", "neutralizing agent", "freeze", "encryption",
    "proof", "molotov cocktail", "rainstorm", "viral meme", "war", "dynamite", "seismic dampener",
    "propaganda", "explosion", "lightning", "evacuation", "flood", "lava", "reforestation", "avalanche",
    "earthquake", "h-bomb", "dragon", "innovation", "hurricane", "tsunami", "persistence", "resilience",
    "terraforming device", "anti-virus nanocloud", "ai kill switch", "nanobot swarm", "reality resynchronizer",
    "cataclysm containment field", "solar deflection array", "planetary evacuation fleet", "antimatter cannon",
    "planetary defense shield", "singularity stabilizer", "orbital laser", "time"
]

words_dict = {
    "sandpaper": [8, 1],
    "oil": [10, 2],
    "steam": [15, 3],
    "acid": [16, 4],
    "gust": [18, 5],
    "boulder": [20, 6],
    "drill": [20, 7],
    "vacation": [20, 8],
    "fire": [22, 9],
    "drought": [24, 10],
    "water": [25, 11],
    "vacuum": [27, 12],
    "laser": [28, 13],
    "life raft": [30, 14],
    "bear trap": [32, 15],
    "hydraulic jack": [33, 16],
    "diamond cage": [35, 17],
    "dam": [35, 18],
    "sunshine": [35, 19],
    "mutation": [35, 20],
    "kevlar vest": [38, 21],
    "jackhammer": [38, 22],
    "signal jammer": [40, 23],
    "grizzly": [41, 24],
    "reinforced steel door": [42, 25],
    "bulldozer": [42, 26],
    "sonic boom": [45, 27],
    "robot": [45, 28],
    "glacier": [45, 29],
    "love": [45, 30],
    "fire blanket": [48, 31],
    "super glue": [48, 32],
    "therapy": [48, 33],
    "disease": [50, 34],
    "fire extinguisher": [50, 35],
    "satellite": [50, 36],
    "confidence": [50, 37],
    "absorption": [52, 38],
    "neutralizing agent": [55, 39],
    "freeze": [55, 40],
    "encryption": [55, 41],
    "proof": [55, 42],
    "molotov cocktail": [58, 43],
    "rainstorm": [58, 44],
    "viral meme": [58, 45],
    "war": [59, 46],
    "dynamite": [60, 47],
    "seismic dampener": [60, 48],
    "propaganda": [60, 49],
    "explosion": [62, 50],
    "lightning": [65, 51],
    "evacuation": [65, 52],
    "flood": [67, 53],
    "lava": [68, 54],
    "reforestation": [70, 55],
    "avalanche": [72, 56],
    "earthquake": [74, 57],
    "h-bomb": [75, 58],
    "dragon": [75, 59],
    "innovation": [75, 60],
    "hurricane": [76, 61],
    "tsunami": [78, 62],
    "persistence": [80, 63],
    "resilience": [85, 64],
    "terraforming device": [89, 65],
    "anti-virus nanocloud": [90, 66],
    "ai kill switch": [90, 67],
    "nanobot swarm": [92, 68],
    "reality resynchronizer": [92, 69],
    "cataclysm containment field": [92, 70],
    "solar deflection array": [93, 71],
    "planetary evacuation fleet": [94, 72],
    "antimatter cannon": [95, 73],
    "planetary defense shield": [96, 74],
    "singularity stabilizer": [97, 75],
    "orbital laser": [98, 76],
    "time": [100, 77]
}

word2id = {word.lower(): idx for idx, word in enumerate(words_list)}
id2word = {idx: word for word, idx in word2id.items()}

# 3. Funcție ultra-rapidă
@torch.no_grad()
def predict(cuvant: str) -> int:
    inputs = tokenizer(cuvant, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logits = model(**inputs).logits.squeeze()

    probs = torch.softmax(logits, dim=-1)
    top3_probs, top3_indices = probs.topk(3)

    best_score = 101
    best_id = 0

    for idx in top3_indices:
        idx = idx.item()
        word = id2word.get(idx, "necunoscut")
        score, word_id = words_dict.get(word, (999, -1))
        if score < best_score:
            best_score = score
            best_id = word_id

    return best_id

# 4. Exemplu de folosire
result = predict("water")
print(f"Rezultatul predicției: {result}")

end_time = time.time()  # Oprim cronometrul

print(f"Timp total de rulare: {end_time - start_time:.4f} secunde")
