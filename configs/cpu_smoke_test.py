# A tiny version of the experiment for checking that the whole pipeline runs on a laptop
# CPU, based on the CPU settings in nanoGPT's README. These runs only show that the code
# works. The model and training are far too small to say anything about the tricks.
dataset = "shakespeare_char"
device = "cpu"
compile_model = False
number_of_layers = 4
number_of_heads = 4
embedding_size = 128
context_length = 64
dropout = 0.0
batch_size = 12
training_steps = 1000
warmup_steps = 50
evaluation_interval = 100
evaluation_batches = 20
sample_length = 200
