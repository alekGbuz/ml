import tensorflow


def multiply_tensors():
    with tensorflow.Graph().as_default():
        a = tensorflow.constant([5, 3, 2, 7, 1, 4])
        b = tensorflow.constant([4, 6, 3])
        reshaped_a = tensorflow.reshape(a, [2, 3])
        reshaped_b = tensorflow.reshape(b, [3, 1])
        multiply_a_b = tensorflow.matmul(reshaped_a, reshaped_b)
        with tensorflow.Session():
            print(multiply_a_b.eval())

if __name__ == "__main__":
    multiply_tensors()
