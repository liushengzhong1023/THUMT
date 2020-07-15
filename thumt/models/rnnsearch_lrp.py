# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils.lrp as lrp

from thumt.models.model import NMTModel


def normalize(matrix, negative=False):
    '''
    Normalize the given matrix at last dimension.
    '''
    if negative:
        matrix_abs = tf.abs(matrix)
        total = tf.reduce_sum(matrix_abs, -1)
        return matrix / tf.expand_dims(total, -1)
    else:
        matrix = tf.abs(matrix)
        total = tf.reduce_sum(matrix, -1)
        return matrix / tf.expand_dims(total, -1)


def stabilize(matrix, stab):
    '''
    Add a small noise at all positions, including 0 positions. Negative noise at negative positions.
    The scale of noise is decided by stab parameter.
    '''
    sign = tf.sign(matrix)
    zero_pos = tf.equal(sign, tf.zeros(tf.shape(sign)))
    zero_pos = tf.cast(zero_pos, tf.float32)
    sign += zero_pos
    result = matrix + stab * sign
    return result


def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)


def _gru_encoder(cell, inputs, sequence_length, initial_state, params,
                 dtype=None):
    '''
    cell: the defined GRU cell
    inputs: [batch, sequence_length, embedding_size]
    initial_state: [batch, rnn_size], may be None
    sequence_length: [batch]
    Define the recurrent structure of one RNN layer.
    '''
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    # zero-state for initial state, a placeholder for state
    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="output_array")

    # Add a w_x_h_ta tensor array here.
    w_x_h_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="w_x_h_array")

    # The initial value of w_x_h is all zero, shape: [batch, time_steps, output_size]
    w_x_h_init = tf.zeros([batch, time_steps, output_size], dtype=tf.float32)

    # Partition each word position to [batch, embedding_size]
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    def loop_func(t, out_ta, state, wxh_ta, w_x_h_last):
        '''
        wxh_ta: saves all wxh vectors.
        w_x_h_last: stores the previous one weight ratio vector.
        '''
        inp_t = input_ta.read(t)

        # w_xlast_newh: the weight ratio from previous hidden state to current hidden state
        # w_x_newh: the weight ratio from current input to current hidden state
        cell_output, new_state, w_xlast_newh, w_x_newh = cell(inp_t, state, w_x_h_last, params)

        # update the weight ratio from input to hidden state, shape [batch, time_steps, output_size]
        # The add operation: replace the zeros in w_xlast_newh for current time step with w_x_newh
        # w_x_h_new stores the weight ratios for all input words to current hidden state
        w_x_newh = tf.pad(w_x_newh, [[0, 0], [t, time_steps - t - 1], [0, 0]])
        w_x_h_new = w_xlast_newh + w_x_newh
        cell_output = _copy_through(t, sequence_length, zero_output, cell_output)
        new_state = _copy_through(t, sequence_length, state, new_state)
        w_x_h_new = _copy_through(t, sequence_length, w_x_h_last, w_x_h_new)
        out_ta = out_ta.write(t, cell_output)
        wxh_ta = wxh_ta.write(t, w_x_h_new)
        return t + 1, out_ta, new_state, wxh_ta, w_x_h_new

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state, w_x_h_ta, w_x_h_init)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    # cell output and state at all word positions.
    output_final_ta = outputs[1]
    final_state = outputs[2]

    # back to [batch, word_sequence, output_size]
    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    # extract all stored weight ratio vectors,
    # shape: [time_steps (for hidden states), batch, time_steps (input positions), output_size]
    w_x_h_final_ta = outputs[3]
    w_x_h_final = w_x_h_final_ta.stack()

    return all_output, final_state, w_x_h_final


def _encoder(cell_fw, cell_bw, inputs, sequence_length, params, dtype=None,
             scope=None):
    with tf.variable_scope(scope or "encoder",
                           values=[inputs, sequence_length]):
        inputs_fw = inputs
        inputs_bw = tf.reverse_sequence(inputs, sequence_length,
                                        batch_axis=0, seq_axis=1)

        # the forward GRU and backward GRU are independent of each other, they are use initial_state=None
        # Compute the weight ratio of input words to forward and backward hidden state.
        with tf.variable_scope("forward"):
            output_fw, state_fw, w_x_h_fw = _gru_encoder(cell_fw, inputs_fw,
                                                         sequence_length, None, params,
                                                         dtype=dtype)

        with tf.variable_scope("backward"):
            output_bw, state_bw, w_x_h_bw = _gru_encoder(cell_bw, inputs_bw,
                                                         sequence_length, None, params,
                                                         dtype=dtype)
            output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                            batch_axis=0, seq_axis=1)

        # add weight_ratios on top of original output
        results = {
            "annotation": tf.concat([output_fw, output_bw], axis=2),
            "outputs": {
                "forward": output_fw,
                "backward": output_bw
            },
            "final_states": {
                "forward": state_fw,
                "backward": state_bw
            },
            "weight_ratios": [w_x_h_fw, w_x_h_bw]
        }

        return results


def _decoder(cell, inputs, memory, sequence_length, initial_state, w_x_enc,
             w_x_bw, params, dtype=None, scope=None):
    '''
    inputs: [batch, sequence length, embedding size]
    memory: the annotation of the encoder, which is [batch, sequence length, 2 * output size]
    sequence_length: [batch], the length of each target sentence
    initial_stage: [batch, hidden size], the last hidden state of encoder.
    w_x_enc: the concatenation of forward and backward layer weight ratio.
             shape: [time_steps (for hidden states), batch, time_steps (input positions), output_size * 2]
    w_x_bw: the backward layer weight ratio.
            shape: [time_steps (for hidden states), batch, time_steps (input positions), output_size]
    Return:
        1) w_x_h: the weight ratio from the input words to the decoder hidden state
        2) w_x_c: the weight ratio from the input words to the context vectors
    '''
    # Assume that the underlying cell is GRUCell-like
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype
    output_size = cell.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)

    with tf.variable_scope(scope or "decoder", dtype=dtype):
        inputs = tf.transpose(inputs, [1, 0, 2])  # [sequence_length, batch, encoder_latent_dim]
        mem_mask = tf.sequence_mask(sequence_length["source"],
                                    maxlen=tf.shape(memory)[1],
                                    dtype=tf.float32)
        bias = layers.attention.attention_bias(mem_mask, "masking")
        bias = tf.squeeze(bias, axis=[1, 2])

        # cache is generated by no query and no bias, used for keys
        cache = layers.attention.attention(None, memory, None, output_size)

        input_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                   tensor_array_name="output_array")
        value_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="value_array")
        alpha_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="alpha_array")
        input_ta = input_ta.unstack(inputs)

        # create tensor array for backward weight ratios
        # w_x_bw_ta: [time_steps (for hidden states), batch, time_steps (input positions), output_size]
        len_src = tf.shape(w_x_bw)[0]
        w_x_bw_ta = tf.TensorArray(tf.float32, len_src, tensor_array_name="w_x_bw_array")
        w_x_bw_ta = w_x_bw_ta.unstack(w_x_bw)

        # shape: [time_steps (for hidden states), batch, time_steps (input positions), 2 * output_size]
        # w_x_c_shape = [batch, time_steps (input positions), output_size]
        w_x_c_shape = tf.shape(w_x_enc)[1:]

        # shape: [batch, time_steps (for hidden states), time_steps (input positions), 2 * output_size]
        w_x_enc = tf.transpose(w_x_enc, [1, 0, 2, 3])

        # shape: [batch, time_steps (for hidden states), time_steps (input positions) * 2 * output_size]
        w_x_enc = tf.reshape(w_x_enc, tf.concat([tf.shape(w_x_enc)[:2], [-1]], -1))

        w_x_h_ta = tf.TensorArray(tf.float32, time_steps, tensor_array_name="w_x_h_array")
        w_x_ctx_ta = tf.TensorArray(tf.float32, time_steps, tensor_array_name="w_x_ctx_array")

        # Linear projection on initial_state
        initial_state_linear = lrp.linear_v2n(initial_state, output_size, True,
                                              [w_x_bw_ta.read(0)], params,
                                              False, scope="s_transform",
                                              d2=True)
        initial_state = initial_state_linear["output"]

        # shape: [batch, time_steps (input positions), decoder_output_size]
        w_initial = initial_state_linear["weight_ratios"][0]
        initial_state = tf.tanh(initial_state)

        def loop_func(t, out_ta, att_ta, val_ta, state, cache_key, wxh_ta, wxc_ta, w_x_h_last):
            '''
            out_ta: output tensor array
            att_ta: attention weight tensor array
            val_ta: weight summed context vector of each output word position
            state: decoder hidden state
            cache_key: cached keys for input word positions.
            wxh_ta: weight ratio from input words to output hidden states.
            wxc_ta: weight ratio from input words to context vectors.
            w_x_h_last: previous hidden state, [batch, time_steps (input positions), decoder_output_size]
            '''
            # now state
            wxh_ta = wxh_ta.write(t, w_x_h_last)
            inp_t = input_ta.read(t)

            # attention layer, return attention weight, and context vector
            # attention layer is the same as original RNNSearch model
            results = layers.attention.attention(state, memory, bias,
                                                 output_size,
                                                 cache={"key": cache_key})

            # alpha: [batch, memory_size], context: [batch, value_size]
            alpha = results["weight"]
            context = results["value"]

            # att: [batch, 1, memory_size]
            att = tf.expand_dims(alpha, 1)

            # weight ratio on attention, [batch, memory_size, 2 * output size]
            wr_att = tf.expand_dims(att, -1) * tf.expand_dims(memory, 1)
            wr_att = tf.squeeze(wr_att, 1)
            result_stab = stabilize(context, params.stab)
            wr_att /= tf.expand_dims(result_stab, 1)

            # length of source sentence = memory_size
            len_src = tf.shape(wr_att)[1]
            w_x_c = tf.reshape(w_x_enc, [1, len_src, len_src, -1]) * wr_att
            w_x_c = tf.reduce_sum(w_x_c, 2)

            # w_x_c: [batch, time_steps (input positions), decoder_output_size]
            w_x_c = tf.reshape(w_x_c, w_x_c_shape)

            # wxc_ta: [output_time_steps (hidden states), batch, time_steps (input positions), decoder_output_size]
            wxc_ta = wxc_ta.write(t, w_x_c)

            # next state, and only preserve elements within source sequence length
            # both w_x_h_last and w_x_c are used as cell input
            # w_x_h shape: [batch, time_steps (input positions), decoder_output_size]
            # w_x_c shape: [batch, time_steps (input positions), decoder_output_size]
            cell_input = [inp_t, context]
            cell_output, new_state, w_x_h_new = cell(cell_input, state, w_x_h_last, w_x_c, params)
            cell_output = _copy_through(t, sequence_length["target"], zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state, new_state)
            new_value = _copy_through(t, sequence_length["target"], zero_value, context)
            w_x_h_new = _copy_through(t, sequence_length["target"], w_x_h_last, w_x_h_new)

            out_ta = out_ta.write(t, cell_output)
            att_ta = att_ta.write(t, alpha)
            val_ta = val_ta.write(t, new_value)
            cache_key = tf.identity(cache_key)

            return t + 1, out_ta, att_ta, val_ta, new_state, cache_key, wxh_ta, wxc_ta, w_x_h_new

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, alpha_ta, value_ta, initial_state, cache["key"], w_x_h_ta, w_x_ctx_ta, w_initial)

        outputs = tf.while_loop(lambda t, *_: t < time_steps,
                                loop_func, loop_vars,
                                parallel_iterations=32,
                                swap_memory=True)

        output_final_ta = outputs[1]
        value_final_ta = outputs[3]

        # output of the cells
        final_output = output_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_output = tf.transpose(final_output, [1, 0, 2])

        # context vectors of the cells
        final_value = value_final_ta.stack()
        final_value.set_shape([None, None, memory.shape[-1].value])
        final_value = tf.transpose(final_value, [1, 0, 2])

        # get wxh_ta and wxc_ta
        # shape: [output_time_steps (hidden states), batch, time_steps (input positions), decoder_output_size]
        w_x_h_final_ta = outputs[6]
        w_x_h_final = w_x_h_final_ta.stack()
        w_x_c_final_ta = outputs[7]
        w_x_c_final = w_x_c_final_ta.stack()

        # initial_state is the last hidden state of the encoder, not changed with decoder
        # w_initial doesn't change with the decoder, initial weight ratios before recurrence
        result = {
            "outputs": final_output,
            "values": final_value,
            "initial_state": initial_state,
            "weight_ratios": [w_x_h_final, w_x_c_final, w_initial]
        }

    return result


def model_graph(features, labels, params):
    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])

    # source and target embedding layer
    with tf.variable_scope("source_embedding"):
        src_emb = tf.get_variable("embedding",
                                  [src_vocab_size, params.embedding_size])
        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])

    with tf.variable_scope("target_embedding"):
        tgt_emb = tf.get_variable("embedding",
                                  [tgt_vocab_size, params.embedding_size])
        tgt_bias = tf.get_variable("bias", [params.embedding_size])
        tgt_inputs = tf.nn.embedding_lookup(tgt_emb, features["target"])

    # add bias for source input and target input
    src_inputs = tf.nn.bias_add(src_inputs, src_bias)
    tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)

    # input dropout
    if params.dropout and not params.use_variational_dropout:
        src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
        tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)

    # encoder, here the cell definition is from lrp file.
    cell_fw = lrp.LegacyGRUCell_encoder_v2n(params.hidden_size)
    cell_bw = lrp.LegacyGRUCell_encoder_v2n(params.hidden_size)

    if params.use_variational_dropout:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            cell_fw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            cell_bw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )

    # apply the encoder
    encoder_output = _encoder(cell_fw, cell_bw, src_inputs,
                              features["source_length"], params)

    w_x_h_fw, w_x_h_bw = encoder_output["weight_ratios"]

    # [::-1] reverse the backward weight ratio list
    # shape: [time_steps (for hidden states), batch, time_steps (input positions), output_size]
    w_x_h_bw = w_x_h_bw[::-1, :, ::-1]

    # concatenate the forward and backward weight ratios
    # shape: [time_steps (for hidden states), batch, time_steps (input positions), output_size * 2]
    w_x_enc = tf.concat([w_x_h_fw, w_x_h_bw], -1)

    # define the decoder RNN cell, also from the LRP file
    cell = lrp.LegacyGRUCell_decoder_v2n(params.hidden_size)

    # dropout on RNN cell if needed
    if params.use_variational_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            # input + context
            input_size=params.embedding_size + 2 * params.hidden_size,
            dtype=tf.float32
        )

    length = {
        "source": features["source_length"],
        "target": features["target_length"]
    }

    # last state of encoder bacward layer
    initial_state = encoder_output["final_states"]["backward"]

    # apply the decoder, feed encoder weight ratio and backward layer weight ratio
    # w_x_enc: [time_steps (for hidden states), batch, time_steps (input positions), output_size * 2]
    # w_x_h_bw: [time_steps (for hidden states), batch, time_steps (input positions), output_size]
    decoder_output = _decoder(cell, tgt_inputs, encoder_output["annotation"],
                              length, initial_state, w_x_enc, w_x_h_bw, params)

    # weight ratio of input words to decoder hidden states, context vectors, and intial states
    # w_x_dec, w_x_ctx shape: [output_time_steps (hidden states), batch, time_steps (input positions), decoder_output_size]
    # w_x_init shape: [batch, time_steps (input positions), decoder_output_size]
    w_x_dec, w_x_ctx, w_x_init = decoder_output["weight_ratios"]

    # Shift left for the target, add one column before the second dimension with value 0
    shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
    shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]

    # all_output shape: [batch, sequence length + 1 (for initial state), output size]
    all_outputs = tf.concat(
        [
            tf.expand_dims(decoder_output["initial_state"], axis=1),
            decoder_output["outputs"],
        ],
        axis=1
    )
    shifted_outputs = all_outputs[:, :-1, :]

    maxout_features = [
        shifted_tgt_inputs,
        shifted_outputs,
        decoder_output["values"]
    ]

    # maxnum is the hidden units of maxout layer, hidden_size is the size of hidden layers
    maxout_size = params.hidden_size // params.maxnum

    # for training and get_relevance, the labels are not none; training uses groundtruth, get_relevance uses predictions
    # for evaluation and inference, the labels are none.
    if labels is None:
        # Special case for non-incremental decoding, only predict the last word
        maxout_features = [
            shifted_tgt_inputs[:, -1, :],
            shifted_outputs[:, -1, :],
            decoder_output["values"][:, -1, :]
        ]

        # maxout layer + linear layer + linear layer (for classification among classes/words)
        maxhid = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                                  params, concat=False)

        readout = layers.nn.linear(maxhid, params.embedding_size, False,
                                   False, scope="deepout")

        # Prediction
        logits = layers.nn.linear(readout, tgt_vocab_size, True, False,
                                  scope="softmax")

        return logits

    # LRP on maxout layer, return a dictionary: 'output' and 'weight_ratios'
    # use w_x_dec and w_x_ctx together
    maxhid_maxout = lrp.maxout_v2n(maxout_features, maxout_size, params.maxnum, [w_x_dec, w_x_ctx], params, concat=False)
    maxhid = maxhid_maxout["output"]
    w_x_maxout = maxhid_maxout["weight_ratios"][0]
    w_x_maxout = tf.transpose(w_x_maxout, [0, 2, 1, 3])

    # first linear layer, return a dictionary: 'output' and 'weight_ratios'
    readout = lrp.linear_v2n(maxhid, params.embedding_size, False, [w_x_maxout], params, False, scope="deepout")
    w_x_readout = readout["weight_ratios"][0]
    readout = readout["output"]

    # dropout on readout
    if params.dropout and not params.use_variational_dropout:
        readout = tf.nn.dropout(readout, 1.0 - params.dropout)

    # Prediction and final relevance
    # w_x_true shape: [batch, input_time_steps, output_time_steps, decoder_output_size]
    logits = lrp.linear_v2n(readout, tgt_vocab_size, True, [w_x_readout], params, False, scope="softmax")
    w_x_true = logits["weight_ratios"][0]
    logits = logits["output"]

    # reshape logits
    logits = tf.reshape(logits, [-1, tgt_vocab_size])

    # shape:
    w_x_true = tf.transpose(w_x_true, [0, 2, 1, 3])

    # shape:
    w_x_true = tf.reshape(w_x_true, [-1, tf.shape(w_x_true)[-2], tf.shape(w_x_true)[-1]])

    # shape:
    w_x_true = tf.transpose(w_x_true, [0, 2, 1])

    # compute labels for LRP, add one column of index before the weight ratios
    # labels_lrp: [index + sequence_length, batch]
    labels_lrp = labels
    bs = tf.shape(labels_lrp)[0]
    idx = tf.range(tf.shape(labels_lrp)[-1])
    idx = tf.cast(idx, tf.int64)
    idx = tf.reshape(idx, [1, -1])
    labels_lrp = tf.concat([idx, labels_lrp], axis=0)
    labels_lrp = tf.transpose(labels_lrp, [1, 0])

    # extract the weight ratios according to the given positions
    # w_x_true shape: [batch, input_time_steps, output_time_steps]
    w_x_true = tf.gather_nd(w_x_true, labels_lrp)
    w_x_true = tf.reshape(w_x_true, [bs, -1, tf.shape(w_x_true)[-1]])

    # compute the cross-entropy loss between predicted word and groundtruth word
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    # tgt_mask specifies the length for each output sentence
    ce = tf.reshape(ce, tf.shape(labels))
    tgt_mask = tf.to_float(
        tf.sequence_mask(
            features["target_length"],
            maxlen=tf.shape(features["target"])[1]
        )
    )

    # store relevance information, only one key: 'result'; relevance = w_x_true?
    rlv_info = {}
    rlv_info["result"] = w_x_true

    # use the same loss for training and get_inference
    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss, rlv_info


class RNNsearchLRP(NMTModel):
    def __init__(self, params, scope="rnnsearch"):
        super(RNNsearchLRP, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=tf.AUTO_REUSE):
                loss = model_graph(features, features["target"], params)[0]
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return evaluation_fn

    def get_relevance_func(self):
        def relevance_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                loss, rlv = model_graph(features, features["target"],
                                        params)
                return features["source"], features["target"], rlv, loss

        return relevance_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return inference_fn

    @staticmethod
    def get_name():
        return "rnnsearch"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="<pad>",
            unk="<unk>",
            eos="<eos>",
            bos="<eos>",
            append_eos=False,
            # model
            rnn_cell="LegacyGRUCell",
            embedding_size=620,
            hidden_size=1000,
            maxnum=2,
            # regularization
            dropout=0.2,
            use_variational_dropout=False,
            label_smoothing=0.1,
            constant_batch_size=True,
            batch_size=128,
            max_length=60,
            clip_grad_norm=5.0,
            # lrp
            stab=0.05
        )

        return params
