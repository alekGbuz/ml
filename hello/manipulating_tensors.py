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
        # idea to get new kind of information replacing columns and rows in initial data
        trn = tf.transpose(a, perm=[0, 2, 1])
        print(trn.get_shape())
        print(trn.eval())
        print("##########")
        print("shape by default")
        trn_default = tf.transpose(a)
        print(trn_default.get_shape())
        print(trn_default.eval())


def space_to_batch_execution():
    # about batch
    # https://stackoverflow.com/questions/41175401/what-is-a-batch-in-tensorflow
    with tf.Graph().as_default(), tf.Session():
        a = tf.constant([[[[1], [2]], [[3], [4]]]])
        print(a.get_shape())
        # from a.get_shape() possible to get data for such things as 4-D with shape [batch, height, width, depth].
        print("##########")
        # paddings in general change indexes by step equal padding/block_size
        # add additional zeros
        # block_size: Non-overlapping blocks of size block_size x block size in
        # the height and width dimensions are rearranged into the batch dimension at each location.
        # (so should save indexes of element from old to new?)
        # use to understand how much parts will be created from initial data
        stb = tf.space_to_batch(a, paddings=[[0, 0], [0, 0]], block_size=2)
        print(stb.get_shape())
        print(stb.eval())
        print("##########")
        # customize out put
        # divides "spatial" dimensions [1, ..., M] of the input into a grid of blocks of shape block_shape
        stbn = tf.space_to_batch_nd(a, block_shape=[1, 2], paddings=[[0, 0], [0, 0]])
        print(stbn.get_shape())
        print(stbn.eval())


def batch_to_space_execution():
    with tf.Graph().as_default(), tf.Session():
        a = tf.constant([[[[1]]], [[[2]]], [[[3]]], [[[4]]]])
        bts = tf.batch_to_space(a, crops=[[0, 0], [0, 0]], block_size=2)
        print(bts.eval())


def dice_simulation():
    # Create a dice simulation, which generates a 10x3 2-D tensor in which:
    # Columns 1 and 2 each hold one throw of one die.
    # Column 3 holds the sum of Columns 1 and 2 on the same row
    with tf.Graph().as_default(), tf.Session() as sess:
        c1 = tf.Variable(tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
        c2 = tf.Variable(tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
        initial = tf.global_variables_initializer()
        sess.run(initial)
        result = tf.concat([tf.concat([c1, c2], 1), tf.add(c1, c2)], 1)
        print(result.eval())


if __name__ == "__main__":
    # multiply_execution()
    # slice_execution()
    # tile_execution()
    # stack_unstack_execution()
    # reverse_execution()
    # transpose_execution()
    # space_to_batch_execution()
    dice_simulation()
