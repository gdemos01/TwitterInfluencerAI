from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# tf.enable_eager_execution() Necessary in Tensorflow 1.X
import numpy as np
import os
import time

class NLP:

    def __init__(self):
        print("> NLP initialized")

    """
        Spliting to training and labeling sets (X,Y)
        @:returns X,Y
    """
    def splitXY(self,chunk):
        X = chunk[:-1]
        Y = chunk[1:]
        return X,Y

    """
        Creating the vocabulary and the coresponding dataset.
        Encoding input text characters to RNN readable numbers
        @:returns dataset, vocabulary
    """
    def preprocess(self,text):
        # Number of Characters in the text input
        print("Length of text: {} characters".format(len(text)))

        # Number of Unique Characters in the text input
        vocabulary = sorted(set(text))
        print('Vocabulary size: {}'.format(len(vocabulary)))

        # Mapping characters to numbers
        self.char_to_index = {u: i for i, u in enumerate(vocabulary)}
        self.index_to_char = np.array(vocabulary)
        text_as_int = np.array([self.char_to_index[c] for c in text])

        # The RNN input sequence of characters length
        seq_length = 100 # 100 characters per sentence

        # Creating Training Sets
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
        dataset = sequences.map(self.splitXY)

        return dataset,vocabulary

    """
        Shuffling the dataset and preparing the model settings
        Play with these settings to improve the performance of the RNN
        @:returns dataset,vocabulary_size,embedding_dimension,rnn_nodes,batch_size 
    """
    def prepareSettings(self,dataset,vocabulary):
        # Preparing the Settings
        batch_size = 64  # Training batch per iteration
        buffer_size = 10000  # Elements in memory
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        vocabulary_size = len(vocabulary)
        embedding_dimension = 256
        rnn_nodes = 1024  # Number of neurons in Recursive Neural Network

        return dataset,vocabulary_size,embedding_dimension,rnn_nodes,batch_size

    """
        Bulding the Deep RNN model
        NOTE:   For different datasets and scenarios I would recommend
                changing the batch_size and considering adding an additional 
                Dropout layer to improve the generalization capabilities and 
                performance of the Deep Model 
        @:returns model
    """
    def buildModel(self,vocabulary_size,embedding_dimension,batch_size,rnn_nodes):
        # Building the Deep Model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocabulary_size, embedding_dimension,
                                      batch_input_shape=[batch_size,None]),
            tf.keras.layers.GRU(rnn_nodes,return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(vocabulary_size)
            # Consider adding an additional Dropout layer to improve performance
        ])
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')

        return model

    """
        Defining training callbacks (Checkpoints here) and training the model
        After each epoch the weights of the RNN are stored in a file (only if
        the loss has decreased)
    """
    def trainModel(self,dataset,model,checkpoint_dir,n_epochs):
        checkpoint_loc = os.path.join(checkpoint_dir,'checkpoint_{epoch}')

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            verbose=1,
            monitor='loss',
            filepath=checkpoint_loc,
            save_best_only=True,
            save_weights_only=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            patience=7,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        )

        model.fit(dataset, epochs=n_epochs, callbacks=[checkpoint_callback,reduce_lr])

    """
        Generating new text (one character at a time) based on an initial seed.
        We make use of the latest checkpoint of the trained Deep RNN model
    """
    def generateText(self,model,checkpoint,seed,text_len):
        # Builds the model based on the last saved checkpoint
        model.load_weights(checkpoint)
        model.build(tf.TensorShape([1,None]))

        # Generating the new Text based on seed
        input_eval = [self.char_to_index[s] for s in seed] # Vectorizing seed
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []

        """Useful note from Tensorflow Tutorial"""
        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 1.0

        # Here batch size == 1
        model.reset_states()
        for i in range(text_len):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.index_to_char[predicted_id])

        return (seed + ''.join(text_generated))



