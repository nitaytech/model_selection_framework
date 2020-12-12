from model_selection_framework.utils import to_iterable, sequence_not_str
import torch
from torch.nn import Sequential, Linear, Dropout
from torch.nn.init import xavier_uniform_
import typing


def init_weights(m: torch.nn.Module):
    if isinstance(m, torch.nn.Linear):
        xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def fc_layers(input_size: int, hidden_sizes: typing.Union[typing.Sequence[int], int],
              output_size: int,
              dropout_p: typing.Union[typing.Sequence[float], float] = None,
              activation_func: typing.Union[typing.Sequence[torch.nn.Module], torch.nn.Module] = None,
              activation_last_layer: bool = False, bias_last_layer: bool = False) -> Sequential:
    """
    create an sequential fully connected linear layer. You should specify the input and output sizes,
     and the hidden layers sizes.
    :param input_size:
    :param hidden_sizes: If a sequence of integers, then this are the hidden layers `input size`.
    If an integer, then the number of hidden layers are equal to this int, and the sizes are exponential decaying
     by 0.5, and won't be smaller than output_size. For e.g. input_size=1000, hidden_sizes=4, output_size=2 then the
     layers sizes are: [1000, 500, 250, 125, 62, 2]
    If a sequence of length of 2, the first element is int and the second is a string,
    the number of hidden layers is equal to the first element (int), and the sizes of the layers are determined
     according to the second element (str):
        * 'equal' - the layers sizes are decreasing by an equal factor of:
         (input_size - output_size) / (hidden_sizes[0] + 1), for e.g. input_size=1000, hidden_sizes=(4, 'equal'),
         output_size=2 then the layers sizes are: [1000, 800, 600, 401, 201, 2] ( -= 199.6)
        * 'exp' - the layers sizes are exponential decaying by factor of:
         (input_size / output_size) ** (1 / (hidden_sizes[0] + 1)), for e.g. input_size=100, hidden_sizes=(4, 'exp'),
         output_size=2 then the layers sizes are: [1000, 288, 83, 24, 6, 2] ( /= 3.466)
    For e.g. input_size=128, hidden_sizes=4 ,output_size=10 then hidden_sizes will be [64, 32, 16, 10]
    Use empty [] or 0 for linear layer without hidden layers.
    :param output_size:
    :param dropout_p: dropout probability [0, 1) between each layer. default None for no dropout.
    :param activation_func: pytorch activation function between each layer.
    :param activation_last_layer: if True use activation function after the last layer.
    :param bias_last_layer: if True use bias in the last layer.
    :return: pytorch Sequential of the linear layers, dropouts and activation functions.
    """
    layers = []

    # hidden_sizes input handler
    if hidden_sizes is None:
        hidden_sizes = []
    elif not sequence_not_str(hidden_sizes) and not isinstance(hidden_sizes, int):
        raise ValueError("hidden_sizes should be a int, a sequence of ints or a pair of [int, 'equal' or 'exp']")
    elif isinstance(hidden_sizes, int):
        hidden_sizes = [max(int(input_size / (2 ** (i + 1))), output_size, 2) for i in range(hidden_sizes)]
    elif sequence_not_str(hidden_sizes) and len(hidden_sizes) == 2 and \
            isinstance(hidden_sizes[0], int) and isinstance(hidden_sizes[1], str):
        if hidden_sizes[1] == 'equal':
            factor = (input_size - output_size) / (hidden_sizes[0] + 1)
            hidden_sizes = [max(int(input_size - factor * (i + 1)), output_size, 2) for i in range(hidden_sizes[0])]
        elif hidden_sizes[1] == 'exp':
            factor = (input_size / output_size) ** (1 / (hidden_sizes[0] + 1))
            hidden_sizes = [max(int(input_size / (factor ** (i + 1))), output_size, 2) for i in range(hidden_sizes[0])]
        else:
            raise ValueError("hidden_sizes should be a int, a list of ints or a pair of [int, 'equal' or 'exp']")
    elif sequence_not_str(hidden_sizes) and \
        sum([1 if isinstance(ele, int) else 0 for ele in hidden_sizes]) == len(hidden_sizes):
        hidden_sizes = list(hidden_sizes)
    else:
        raise ValueError("hidden_sizes should be a int, a list of ints or a pair of [int, 'equal' or 'exp']")
    # after validating the hidden_sizes input
    sizes = [input_size] + hidden_sizes + [output_size]
    n_layers = len(sizes) - 1
    # activation_func input handler
    if not sequence_not_str(activation_func):
        activations = [activation_func] * n_layers
    else:
        if len(activation_func) != n_layers:
            raise ValueError(f"len(activation_func) should be equal to number of layers."
                             f" Got {len(activation_func)} and {n_layers}")
        activations = list(activation_func)
    # dropout_p input handler
    if not sequence_not_str(dropout_p):
        dropouts = [dropout_p] * n_layers
    else:
        if len(dropout_p) != n_layers:
            raise ValueError(f"len(dropout_p) should be equal to number of layers."
                             f" Got {len(dropout_p)} and {n_layers}")
        dropouts = list(dropout_p)
    # creating the fully connected layers
    for i in range(1, n_layers + 1):
        single_layer = []
        bias = False if (i == n_layers and not bias_last_layer) else True  # last layer should not have bias,
        # otherwise net converges to mean prediction
        single_layer.append(Linear(in_features=sizes[i - 1], out_features=sizes[i], bias=bias))
        dropout, activation = dropouts[i-1], activations[i-1]
        if dropout is not None and dropout > 0:
            single_layer.append(Dropout(p=dropout))
        if activation is not None and (
                i < n_layers or activation_last_layer):  # no non-linear activation in the last layer
            single_layer.append(activation())
        layers.append(Sequential(*single_layer))
    return Sequential(*layers)


class Attention(torch.nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:
            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    Example:
         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class AggAttention(torch.nn.Module):
    def __init__(self, input_size):
        super(AggAttention, self).__init__()
        self.context = torch.nn.Parameter(torch.normal(mean=0, std=1, size=(input_size, )), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query):
        x = torch.matmul(query, self.context)
        x = self.softmax(x)
        x = torch.mul(query, x.unsqueeze(-1))
        return torch.sum(x, dim=1)

