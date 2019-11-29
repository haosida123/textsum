# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Beam search module.

Beam search takes the top K results from the model, predicts the K results for
each of the previous K result, getting K*K results. Pick the top K results from
K*K results, and start over again until certain number of results are fully
decoded.
"""

import tensorflow as tf
import numpy as np


class Hypothesis(object):
    """Defines a hypothesis during beam search."""

    def __init__(self, tokens, log_prob, state):
        """Hypothesis constructor.

        Args:
          tokens: start tokens for decoding.
          log_prob: log prob of the start tokens, usually 1.
          state: decoder initial states.
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state

    def Extend(self, token, log_prob, new_state):
        """Extend the hypothesis with result from latest step.

        Args:
          token: latest token from decoding.
          log_prob: log prob of the latest decoded tokens.
          new_state: decoder output state. Fed to the decoder for next step.
        Returns:
          New Hypothesis with the results from latest step.
        """
        return Hypothesis(self.tokens + [token], self.log_prob + log_prob,
                          new_state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob,
                                                              self.tokens))


class BeamSearch(object):
    """Beam search."""

    def __init__(self, model, beam_size, start_token, end_token, max_steps, normalize_by_length=True):
        """Creates BeamSearch object.

        Args:
          model: Seq2SeqAttentionModel.
          beam_size: int.
          start_token: int, id of the token to start decoding with
          end_token: int, id of the token that completes an hypothesis
          max_steps: int, upper limit on the size of the hypothesis
        """
        self._model = model
        self._beam_size = beam_size
        self._start_token = start_token
        self._end_token = end_token
        self._max_steps = max_steps
        self.normalize_by_length = normalize_by_length

    # @tf.function
    def beam_search(self, enc_inputs):
        """Performs beam search for decoding.

        Args:
          enc_inputs: ndarray of shape (enc_length,), the document ids to encode

        Returns:
          hyps: list of Hypothesis, the best hypotheses found by beam search,
              ordered by score
        """

        # Run the encoder and extract the outputs and final state.
        enc_hidden = self._model.encoder.initialize_hidden_state(batch_size=1)
        enc_top_states, dec_in_state = self._model.encoder(
            tf.expand_dims(enc_inputs, 0), enc_hidden)
        # Replicate the initial states K times for the first step.
        enc_top_states = tf.tile(enc_top_states, [self._beam_size, 1, 1])
        hyps = [Hypothesis([self._start_token], 0.0, dec_in_state)
                ] * self._beam_size
        results = []
        steps = 0
        while steps < self._max_steps and len(results) < self._beam_size:
            latest_tokens = tf.expand_dims([h.latest_token for h in hyps], 1)
            states = tf.concat([h.state for h in hyps], axis=0)
            # states = tf.convert_to_tensor([h.state for h in hyps])
            # outputs: (batch_size, vocab_size)
            # new_states: (batch, dec units)
            # print([t.shape for t in [latest_tokens, states, enc_top_states]])
            outputs, new_states, _ = self._model.decoder(
                latest_tokens, states, enc_top_states)
            topk_ids = tf.argsort(
                outputs, axis=1)[:, ::-1][:, :self._beam_size * 2]
            topk_log_probs = tf.math.log(tf.sort(
                outputs, axis=1)[:, ::-1][:, :self._beam_size * 2])
            # # id_prob_state = sorted(
            # #     [(idx, np.log(lo), state) for (idx, lo, state) in
            # #      zip([range(len(outputs)), outputs, new_states])
            # #      ], key=lambda x: x[1], reverse=True
            # # )[-self._beam_size:]
            # # topk_ids, topk_log_probs, new_states = [[
            # #     ips[i] for ips in id_prob_state] for i in range(3)]
            # Extend each hypothesis.
            all_hyps = []
            # The first step takes the best K results from first hyps. Following
            # steps take the best K results from K*K hyps.
            num_beam_source = 1 if steps == 0 else len(hyps)
            for i in range(num_beam_source):
                h, ns = hyps[i], tf.expand_dims(new_states[i], 0)
                for j in range(self._beam_size * 2):
                    all_hyps.append(
                        h.Extend(topk_ids[i, j], topk_log_probs[i, j], ns))
                    # h.Extend(token, log_prob, new_state)

            # Filter and collect any hypotheses that have the end token.
            hyps = []
            for h in self._BestHyps(all_hyps):
                if h.latest_token == self._end_token:
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(h)
                else:
                    # Otherwise continue to the extend the hypothesis.
                    hyps.append(h)
                if len(hyps) == self._beam_size or len(results) == self._beam_size:
                    break

            steps += 1

        if steps == self._max_steps:
            results.extend(hyps)

        return self._BestHyps(results)

    # @tf.function
    def _BestHyps(self, hyps):
        """Sort the hyps based on log probs and length.

        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        # This length normalization is only effective for the final results.
        if self.normalize_by_length:
            ret = [h.log_prob for h in hyps]
        else:
            ret = [h.log_prob / (len(h.tokens) + 1) for h in hyps]
        # tf.print(type(ret))
        # assert(type(ret) == tf.Tensor)
        idx = tf.argsort(tf.convert_to_tensor(ret), direction='DESCENDING')
        return [hyps[i] for i in idx]
        # if self.normalize_by_length:
        #     return sorted(hyps, key=lambda h: h.log_prob / (len(h.tokens) + 1), reverse=True)
        #     # return tf.sort(hyps, key=lambda h: h.log_prob / (len(h.tokens) + 1), direction='DESCENDING')
        # else:
        #     return sorted(hyps, key=lambda h: h.log_prob, reverse=True)
        #     # return tf.sort(hyps, key=lambda h: h.log_prob, direction='DESCENDING')

    def print_beam_search(self, enc_inputs, vocab, targets=None):
        hyps = self.beam_search(enc_inputs)
        print('Beam search inputs:\t' + ''.join(
            [vocab.to_tokens(w) for w in enc_inputs if w not in [
                vocab.pad]
                # vocab.bos, vocab.eos, vocab.pad]
             ]).replace('seperator', ','))
        if targets is not None:
            print('Target:\t{}'.format(
                ''.join([vocab.to_tokens(t) for t in targets if t not in [vocab.bos, vocab.eos, vocab.pad]]
                        ).replace('seperator', ',')))
        print('Predicted:')
        for hyp in hyps[:5]:
            print('\t\tprob:{:.4e}'.format(tf.exp(hyp.log_prob)))
            print('\t\t{}'.format(
                ''.join([vocab.to_tokens(t) for t in hyp.tokens if t not in [vocab.pad, vocab.unk]]).replace('seperator', ',')))
                # ''.join([vocab.to_tokens(t) for t in hyp.tokens if t not in [vocab.bos, vocab.eos, vocab.pad, vocab.unk]]).replace('seperator', ',')))
