import tensorflow as tf


def multiply_execution():
    with tf.Graph().as_default():
        a = tf.constant([5, 3, 2, 7, 1, 4])
        b = tf.constant([4, 6, 3])
        reshaped_a = tf.reshape(a, [2, 3])
        reshaped_b = tf.reshape(b, [3, 1])
        multiply_a_b = tf.matmul(reshaped_a, reshaped_b)
        with tf.Session():
            print(multiply_a_b.eval())


def slice_execution():
    with tf.Graph().as_default(), tf.Session():
        t = tf.constant([
                [[1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4]],
                [[5, 5, 5], [6, 6, 6]]])
        slc = tf.slice(t, [1, 0, 0], [1, 1, 3])
        print(slc.eval())
        print("##########")
        slc = tf.slice(t, [1, 0, 0], [1, 2, 3])
        print(slc.eval())
        print("##########")
        slc = tf.slice(t, [1, 0, 0], [2, 1, 3])
        print(slc.eval())
        print("##########")
        slc = tf.slice(t, [0, 0, 0], [3, 1, 1])
        print(slc.eval())


def tile_execution():
    with tf.Graph().as_default(), tf.Session():
        t = tf.constant([1, 2, 4])
        tilling_value = tf.constant([2])
        tl = tf.tile(t, tilling_value)
        print(tl.eval())


def stack_unstack_execution():
    with tf.Graph().as_default(), tf.Session() as sess:
        x = tf.constant([1, 4])
        y = tf.constant([2, 5])
        z = tf.constant([3, 6])
        stc = tf.stack([x, y, z])
        print(stc.eval())
        print("##########")
        ustc = tf.unstack(stc)
        print(sess.run(ustc))


def reverse_execution():
    with tf.Graph().as_default(), tf.Session():
        a = tf.constant(
                [[[[0,  1,  2,  3],
                 [4,  5,  6,  7],
                 [8,  9, 10, 11]],
                 [[12, 13, 14, 15],
                  [16, 17, 18, 19],
                  [20, 21, 22, 23]]]])
        print(a.get_shape())
        rvs = tf.reverse(a, [1, 3])
        print(rvs.eval())


def transpose_execution():
    with tf.Graph().as_default(), tf.Session():
        a = tf.constant([[
            [1,  2,  3],
            [4,  5,  6]],
            [[7,  8,  9],
             [10, 11, 12]]])
        # initial shape for a is (2,2,3)->using perm=[0, 2, 1] should have new shape for tensor (2,3,2)
        # perm show what dimensions should replace by id during transpose
        # e.g. 3 had 2 index but has 1 in new tensors shape
        # idea to get new kind of information replacing colums and rows in initial data
        trn = tf.transpose(a, perm=[0, 2, 1])
        print(trn.get_shape())
        print(trn.eval())
        print("##########")
        print("shape by default")
        trn_default = tf.transpose(a)
        print(trn_default.get_shape())
        print(trn_default.eval())


def dice_simulation():
    pass

if __name__ == "__main__":
    # multiply_execution()
    # slice_execution()
    # tile_execution()
    # stack_unstack_execution()
    # reverse_execution()
    transpose_execution()
