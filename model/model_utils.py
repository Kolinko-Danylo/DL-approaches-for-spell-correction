
from semichar_rnn import SemiCharRNN

def get_model(model_name, dataset_params, n_hidden, n_layers, drop_prob, grad_clip):
  if model_name == "semichar_rnn":
    net = SemiCharRNN(dataset_params, n_hidden, n_layers, drop_prob, grad_clip)
  elif model_name == "seq2seq+attention":
    return
  else:
    raise ValueError(f"Architecture with such name ({model_name}) is nt defined")
  return net