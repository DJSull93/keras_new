from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'
# {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}
# text -> 3 1 4 5 6 1 2 2 7
# needs to do OneHotEncoding

# indexing number by frquant, order 
token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)

