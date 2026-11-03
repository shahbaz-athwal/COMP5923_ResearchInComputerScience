# The main experiment: nanoGPT's Shakespeare-char settings. Most of these are already the
# defaults in ablation/settings.py, but I write them out so the experiment is explicit.
dataset = "shakespeare_char"
number_of_layers = 6
number_of_heads = 6
embedding_size = 384
context_length = 256
dropout = 0.2
batch_size = 64
training_steps = 5000
learning_rate = 1e-3
warmup_steps = 100
adam_beta2 = 0.99
evaluation_interval = 250
evaluation_batches = 50  # only for the loss curves; the final number uses the whole validation split
compile_model = True
