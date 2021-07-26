from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', '참 재밌네요', '청순이가 잘 생기긴 했어요'
        ]

# 긍정 1 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])
token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
'''
{'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, 
'추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, 
'싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 
20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '청순이가': 25, 
'생기긴': 26, '했어요': 27}
'''

x = token.texts_to_sequences(docs)
# print(x)
'''
## size not same 
[[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
앞부터 0으로 채워서 사이즈 맞춤 (like padding)
-> 뒤의 데이터가 예측값에 가장 큰 가중치
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
# different size data -> put 0 in lesser size vector
pad_x = pad_sequences(x, padding='pre', maxlen=5) # 'pre' <> 'post'
'''
maxlen > numb -> 너무 큰 데이터 들어오면 0 많아짐, 더 짧게 자를 수 있고,
자르게 되면 앞부터 데이터 짤림
'''
print(pad_x)
print(pad_x.shape) # (13, 5)

# 원핫인코딩 -> 데이터 수 기하급수적으로 늘어남
# 