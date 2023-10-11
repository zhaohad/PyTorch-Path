import torch

path = "./jane-austen/1342-0.txt"

with open(path, encoding='utf8') as f:
    text = f.read()

lines = text.split('\n')

print(f"len(lines) = {len(lines)}")

line = lines[200]

print(f"line200 = {line}")

letter_t = torch.zeros(len(line), 128)

print(f"letter_t.shape = {letter_t.shape}")

for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1

print(f"letter_t = {letter_t}")

