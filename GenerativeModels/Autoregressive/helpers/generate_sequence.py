import torch
import torch.distributions as dist
import torch.nn.functional as F

# This blog post shows how temperature sampling works
# https://lukesalamone.github.io/posts/what-is-temperature/#:~:text=Temperature%20is%20a%20parameter%20used,to%20play%20with%20it%20yourself.
def temp_sampling(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

def generate_sequence(model, seed, seq_length, generate_length=600, temperature=0.5, verbose=False):
    """
    Sequentially samples a sequence from the model, token by token.
    :param model:
    :param seed: The sequence to start with.
    :pararm seq_length: other names token_nunber / max_seq_length
    :param generate length: The total number of characters to be generated.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.
    :return: The sampled sequence, including the seed.
    """

    sequence = seed.detach().clone()

    if verbose: # Print the seed, surrounded by square brackets
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(generate_length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-seq_length:]  #!!! this is the important trick

        # Run the current input through the model 
        # we treat the data as a single batch entery to the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = temp_sampling(output[0, -1, :], temperature) #!!! sampling form the last output seq 

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0) # !!! Append the sampled token to the sequence

    print()
    return seed