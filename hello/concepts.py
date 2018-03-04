import tensorflow

# Create a graph.
g = tensorflow.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
    x = tensorflow.constant(8, name="x_const")
    y = tensorflow.constant(5, name="y_const")
    z = tensorflow.constant(4, name="z_const")
    x_y_sum = tensorflow.add(x, y, name="x_y_sum")
    x_y_z_sum = tensorflow.add(x_y_sum, z, name="x_y_z_name")

    # Now create a session.
    # The session will run the default graph.
    with tensorflow.Session() as sess:
        # If t is a Tensor object, t.eval() is shorthand for sess.run(t) (where sess is the current default session.)
        print(x_y_z_sum.eval())


