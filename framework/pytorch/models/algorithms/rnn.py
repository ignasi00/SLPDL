
import torch
import torch.nn as nn


class RNNClassifier(torch.nn.Module):
    # The algorithm consist on applying a unique expresion (recurrent_unit) recurrently; it is a kind of IIR filter

    def __init__(output_size, representation_model, recurrent_unit, recurrent_output_size, bidirectional=False, dropout=0, pooling='max'):
        super().__init__()
        self.pooling = pooling.lower()

        self.embed = representation_model
        # Reinventando la rueda: dada una recurrent unit (puede tenga N layers y tenga la forma que sea), aplicar M veces
        # Input: Tensor(T x B' x E) State(L x B' x H)[default: 0]; Output: Tensor(T x B' x H) State(L x B' x H)
        self.rnn = recurrent_unit

        self.dropout = nn.Dropout(p=dropout)

        if pooling in ['max', 'mean', 'add']:
            self.h2o = nn.Linear(recurrent_output_size, output_size)
        elif pooling == 'cat':
            self.h2o = nn.Linear(2 * recurrent_output_size, output_size)
        else:
            raise Exception('pooling type not contemplated on the model')
    
    def forward(self, input_, input_lengths=None):
        if input_lengths is None:
            if isinstance(input_, (list, tuple)):
                input_lengths = torch.tensor([x.numel() for x in input_], dtype=torch.long)
                # 1 x B
                input_ = torch.nn.utils.rnn.pad_sequence(input_).to(self.h2o.weight.device)
                # T x B
            else:
                input_lengths = torch.tensor([input_.shape[0]] * input_.shape[1], dtype=torch.long)
                # 1 x B

        encoded = self.embed(input_)
        # T x B x E

        packed = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_lengths, enforce_sorted=False)
        # Packed T x B x E; data (flatten B(T) x E), batch_sizes (values for each time), sorted_indices (long to short indices), unsorted_indices (sorted_indices -> range)

        state = None
        output = list()
        for idx, current_input in enumerate(encoded):

            current_indxs = packed.sorted_indices[:packed.batch_sizes[idx]]
            if state is not None:
                state_idxs = current_indxs - sum([(current_indxs > x) for x in packed.sorted_indices[packed.batch_sizes[idx]:]])
                state = state[state_idxs]
            output_rnn, state = self.rnn(current_input[current_indxs], state)

            output.append(output_rnn)
        output = PackedSequence(torch.cat(output), packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)
        # Packed T x B x H

        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=float('-inf'))
        # T x B x H
               
        if MODE == 'max':
            output, _ = padded.max(dim=0)
            # B x H
        elif MODE == 'mean':
            padded = torch.nan_to_num(padded, neginf=0.0)
            output = padded.mean(dim=0, keepdim=False)
            # B x H
        elif MODE == 'add':
            padded_mean = torch.nan_to_num(padded, neginf=0.0)
            output = padded.max(dim=0)[0] + padded_mean.mean(dim=0, keepdim=False)
            # B x H
        elif MODE == 'cat':
            padded_mean = torch.nan_to_num(padded, neginf=0.0)
            output = torch.cat([padded.max(dim=0)[0], padded_mean.mean(dim=0, keepdim=False)], dim=-1)
            # B x 2*H <-- At constructor the 2*H is solved
        
        output = self.dropout(output)
        output = self.h2o(output)
        # B x O
        
        return output
