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


def clean_words(input_str):
    punctuation = '.,;:\"!?”“_-'
    word_list = input_str.lower().replace('\n', ' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list


words_in_line = clean_words(line)
print(f"words_in_line = {words_in_line}")

word_list = sorted(set(clean_words(text)))
print(f"word_list = {word_list}")

word2index_dict = {word: i for (i, word) in enumerate(word_list)}
print(f"word2index_dict = {word2index_dict}")

word_t = torch.zeros(len(words_in_line), len(word2index_dict))
print(f"word_t.shape = {word_t.shape}")
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
    print('{:2} {:4} {}'.format(i, word_index, word))
