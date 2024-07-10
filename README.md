A continuation of the backpropagation and makemore project, GPT is trying to emulate the Chat-GPT model from OpenAI in a much smaller and not as time-sensitive manner. In order to do this, let's try to create a character tokenization method that is similar to ChatGPT's tokenization (but smaller!), and make a simpler but just as powerful version as OpenAI's ChatGPT-2 model! 

In order to do this, let's analyze some papers and some important realizations that help us to make this model:
    1) First, a baseline setup is inspired by the paper <a href = "https://arxiv.org/pdf/1706.03762" target = "_blank">"Attention is All You Need"</a>, which emphasizes the need of self-attention and parallel attention in order to create blocks in which values, keys, and queries can talk to each other to determine values like the consonants, vowels, and words that we use in LLMs. 