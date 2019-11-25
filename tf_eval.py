from preprocessing import replace_sentence, tokenize_sentence
import tensorflow as tf


def evaluate(encoder, decoder, sentence, vocab, max_length_inp, max_length_targ):

    sentence = tokenize_sentence(replace_sentence(sentence))

    inputs = [vocab[w] for w in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, encoder.enc_units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([vocab.bos], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, _ = decoder(
            dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += vocab[predicted_id] + ' '

        if vocab[predicted_id] == vocab.eos:
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def predict(sentence):
    result, sentence = evaluate(
        encoder, decoder, sentence, vocab, max_length_inp, max_length_targ)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))
