import numpy as np

def find_closest_vector(n, input_vector, known_vectors, error_type='SE'):
    if len(input_vector) == 0:
    	return -1
    """
    找出與輸入向量最相近的已知向量索引（根據誤差最小）。
    在誤差計算時，考慮角度環狀特性（例如0度和179度之間的差為1度）。

    參數：
        n (int): 維度數。
        input_vector (list or np.ndarray): 輸入的 n 維向量。
        known_vectors (list of list or np.ndarray): 10 個已知的 n 維向量。
        error_type (str): 比對方法，'AE' 為絕對誤差，'SE' 為平方誤差。

    回傳：
        int: 誤差最小的向量在 known_vectors 中的索引。
    """
    print("2")
    input_vector = np.array(input_vector)
    print("3")
    known_vectors = np.array(known_vectors)
    print("4")
    for i in input_vector:
        print(f"{i}, ", end="")

    if input_vector.shape[0] != n or known_vectors.shape[1] != n:
        print(f"input vector shape = {input_vector.shape[0]}")
        print(f"known vector shape = {known_vectors.shape[1]}")
        # raise ValueError("輸入向量或已知向量的維度與 n 不符")

    # 計算差值並考慮360度內的最小誤差
    diff = np.abs(known_vectors - input_vector)
    wrapped_diff = np.minimum(diff, 360 - diff)

    if error_type == 'AE':
        errors = np.sum(wrapped_diff, axis=1)
    elif error_type == 'SE':
        errors = np.sum(wrapped_diff ** 2, axis=1)
    else:
        raise ValueError("錯誤類型必須是 'AE' 或 'SE'")

    print(errors)
    min_index = np.argmin(errors)
    return min_index
'''
#改成動作名稱
action=[
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J"
]

#模擬情境
n = 12
input_vector = [305, 242, 250, 115, 190, 266, 274, 0, 258, 251, 285, 283] #輸入的向量
known_vectors = [  #多個動作的向量
    [303, 242, 250, 115, 192, 266, 274, 5, 258, 251, 285, 283],
    [38, 145, 160, 74, 184, 266, 275, 353, 270, 262, 269, 270],
    [265, 210, 233, 99, 204, 277, 280, 0, 255, 278, 247, 285],
    [281, 121, 225, 213, 170, 265, 272, 358, 256, 253, 285, 279],
    [283, 62, 259, 235, 151, 261, 265, 312, 263, 261, 257, 258],
    [310, 101, 236, 92, 174, 268, 275, 6, 261, 265, 283, 283],
    [356, 104, 246, 74, 187, 268, 273, 357, 267, 271, 269, 262],
    [93, 73, 242, 254, 176, 257, 267, 340, 272, 260, 265, 259],
    [272, 123, 270, 85, 174, 265, 273, 4, 252, 285, 270, 263],
    [76, 142, 96, 45, 185, 267, 272, 1, 268, 277, 283, 255]
]

index = find_closest_vector(n, input_vector, known_vectors, error_type='SE')
print(f"動作判定：{action[index]}")
'''
