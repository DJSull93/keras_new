'''
텍스트 데이터 학습을 위해 텍스트를 연산 가능한 정수로 변환
Tokenizer : 빈도 많은 단어 우선, 이후 순차적으로 단어 정수로 변환
'''

from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'
# text -> 3 1 4 5 6 1 2 2 7
# needs to do OneHotEncoding

# indexing number by frquant, order 
token = Tokenizer()
token.fit_on_texts([text])

# print(token.word_index) # {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}

x = token.texts_to_sequences([text])
print(x) # [[3, 1, 4, 5, 6, 1, 2, 2, 7]]

from tensorflow.keras.utils import to_categorical

word_size = len(token.word_index)
print(word_size) # 7

x = to_categorical(x)
print(x.shape) # 1, 9, 8 -> word 9, to_cat 0, 1~7 : 8
# 라벨의 갯수만큼 열 생성