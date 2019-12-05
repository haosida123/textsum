# from preprocessing import replace_sentence, tokenize_sentence
# from utils.config import params
# from seq2seq_att import Encoder, Decoder
import paddle.fluid as fluid

# from data import TextDataset


class Evaluate():
    def __init__(self):
        self.result = tf.Variable([0])

    @tf.function
    def __call__(self, inputs, begin_id, end_id, max_length_targ, encoder, decoder):
        inputs = tf.convert_to_tensor(inputs)
        result = self.result
        result.assign([begin_id])
        hidden = encoder.initialize_hidden_state(inputs=inputs)
        self.enc_out, enc_hidden = encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.cast(tf.expand_dims([begin_id], 0), tf.int32)
        end, t = False, tf.cast(0, tf.int32)

        def cond(t, end, dec_hidden, dec_input, result):
            return t < max_length_targ and not end

        def loop(t, end, dec_hidden, dec_input, result):
            predictions, dec_hidden, _ = decoder(
                dec_input, dec_hidden, self.enc_out)
            predicted_id = tf.cast(tf.math.argmax(predictions, axis=1), tf.int32)  # .numpy()
            result = tf.concat([result, predicted_id], 0)
            # tf.print(result, predicted_id)
            if predicted_id == end_id:
                end = True
            else:
                end = False
            # the predicted ID is fed back into the model
            dec_input = tf.cast(tf.expand_dims(predicted_id, 0), tf.int32)
            # dec_input = tf.cast(tf.expand_dims([predicted_id], 0), tf.int32)
            return (tf.cast(t+1, tf.int32), end, dec_hidden, dec_input, result)
        _, _, _, _, result = tf.while_loop(cond, loop, loop_vars=[t, end, dec_hidden, dec_input, result],
                      shape_invariants=[t.get_shape(),
                                        tf.TensorShape(None),
                                        dec_hidden.get_shape(),
                                        dec_input.get_shape(),
                                        tf.TensorShape([None])])
        return result


def predict_array_input(inputs, vocab, max_length_targ, encoder, decoder, targets=None, beam_search=None):
    # inputs = TextDataset.build_array([inputs], vocab, max_length_inp, is_source=True)[0]
    evaluate = Evaluate()
    if beam_search is not None:
        beam_search.print_beam_search(inputs, vocab, targets)
        result = evaluate(
            [inputs], vocab.bos, vocab.eos, max_length_targ, encoder, decoder
        )
        print('Non-beam-search Predicted:\n\t{}'.format(
            ''.join([vocab.to_tokens(t) for t in result if t not in [vocab.pad, vocab.unk]]).replace('seperator', ',')))
        # ''.join([vocab.to_tokens(t) for t in result if t not in [vocab.bos, vocab.eos, vocab.pad, vocab.unk]]).replace('seperator', ',')))
    else:
        print('Input:\t%s' % ''.join([vocab.to_tokens(w) for w in inputs if w not in [
              vocab.bos, vocab.eos, vocab.pad]]).replace('seperator', ','))
        result = evaluate([inputs], vocab.bos, vocab.eos,
                          max_length_targ, encoder, decoder)
        print('Predicted:\t{}'.format(''.join([vocab.to_tokens(t) for t in result if t not in [
              vocab.bos, vocab.eos, vocab.pad, vocab.unk]]).replace('seperator', ',')))
        if targets is not None:
            print('Target:\t{}'.format(''.join([vocab.to_tokens(
                t) for t in targets if t not in [vocab.bos, vocab.eos, vocab.pad]])))
