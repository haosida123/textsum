from preprocessing import replace_sentence, tokenize_sentence
# from utils.config import params
# from seq2seq_att import Encoder, Decoder
import tensorflow as tf

from data import TextDataset


def evaluate(inputs, vocab, max_length_targ, encoder, decoder):

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

        result += vocab.to_tokens(predicted_id) + ' '

        if predicted_id == vocab.eos:
            return result

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def predict(sentence, vocab, max_length_inp, max_length_targ, encoder, decoder):
    sentence = tokenize_sentence(replace_sentence(sentence))
    inputs = [vocab[w] for w in sentence]
    # inputs = tf.keras.preprocessing.sequence.pad_sequences(
    #     [inputs], maxlen=max_length_inp, padding='post')
    inputs = TextDataset.build_array(
        [inputs], vocab, max_length_inp, is_source=True)[0]
    result = evaluate(
        inputs, vocab, max_length_targ, encoder, decoder)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))


def predict_array_input(inputs, vocab, max_length_targ, encoder, decoder, targets=None):
    # inputs = TextDataset.build_array([inputs], vocab, max_length_inp, is_source=True)[0]
    result = evaluate(
        [inputs], vocab, max_length_targ, encoder, decoder)

    print('Input: %s' % (vocab.to_tokens(inputs)))
    print('Predicted translation: {}'.format(result))
    if targets is not None:
        print('Target translation: {}'.format(
            ''.join([vocab.to_tokens(t) for t in targets if t not in [vocab.bos, vocab.eos, vocab.pad]]
                    ).replace('seperator', ',')))
