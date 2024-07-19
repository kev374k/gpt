# An introduction to a re-replication of ChatGPT 2.0
A continuation of the backpropagation and makemore project, GPT is trying to emulate the Chat-GPT model from OpenAI in a much smaller and not as time-sensitive manner. In order to do this, let's try to create a character tokenization method that is similar to ChatGPT's tokenization (but smaller!), and make a simpler but just as powerful version as OpenAI's ChatGPT-2 model! 

In order to do this, let's analyze some papers and some important realizations that help us to make this model:
1. First, a baseline setup is inspired by the paper <a href = "https://arxiv.org/pdf/1706.03762" target = "_blank">"Attention is All You Need"</a>, which emphasizes the need of self-attention and parallel attention in order to create blocks in which values, keys, and queries can talk to each other to determine values like the consonants, vowels, and words that we use in LLMs. We also added position-wise feed-forward networks and positional encoding in order to make sure the model is smarter and can generate better results.
2. Second, a paper by OpenAI titled <a href = "https://arxiv.org/pdf/2005.14165" target = "_blank">Lnaguage Models are Few-Shot Learners</a> which gives us the intrinsic differences between LLMs like ChatGPTs and other pretrained models (that used to be the norm), and why GPTs specifically have become so popular and so ready-to-use in the industry.

Introduction:
GPTs are generally transformers that are based on "in-context" learning, such that they are mainly trained one time on massive scales of data in order to be good at essentially everything. For example, GPT-3 was trained such that it developed over 175 billion parameters, and this is contrasted with other fine-tuned models, which have specific datasets specific to desired tasks. GPTs are not inherently fine tuned, because it is expensive and requires new large datasets for every task, making it non task-agnostic. GPTs are generally designed to be able to answer questons based on a broad spectrum of topics, but their categories are separated as such:
1. Zero-shot: The model predicts the answer given only a natural language description of the task.
2. One-shot: In additon to the task description, the model sees a single xample of the task
3. Few-shot: In addition to the task description, the model sees a few examples of the task. 

This approach allows for "maximum convenience, potential for robustness, and avoidance of spurious correlations, but is also the most challenging setting". This is because the ambiguousness of some tasks can even confuse humans, and it can be unclear as to what the user is actually searching for (at least in reference to the zero-shot approach above). 

Usage:
1. Based on the desired use-case, this model can perform up to GPT-2 standards when using a set of data like the world wide web. However, this takes a very long time of running, even with only up to 5000 steps. In order to train the baseline model that high, you would need approximately 3-4 days in total with CUDA to perform that well. If you are using something like a MacBook (which lacks a NVIDIA GPU), then instead this will take significantly longer. Adjust the model parameters (at the top of "gpt_final_version.py") to make the batch size, block_size, embedding parameters, head and layer parameters smaller so that there will be a faster result. Additionally, if you are using a M-series MacBook, use the dedicated device "mps" to improve performance. 
2. Additionally, there are extra imports at the top of the page. All the PyTorch modules are self-explanatory, but we use the dataclass module in order to set-up a model configuration that would work for the the rest of the classes. 
3. With the way the current parameters are set-up, I got my model to train through 5000 steps in approximately ~ 3 hours and 15 minutes
4. In order to save the model, use the ModelHandler() class and initialize a file save path in order to save the model. Similarly, to load a model in the future, you can use the ModelHandler() class to load it and train it more, or use it to generate more tokens. Some examples of models are included in the gpt/models folder, where 3 examples are located. Notes of how the models are initialized are within the gpt/notes.txt document, where model2 had the most extensive training parameters, received a validation loss of approximately 1.39, and had over 3.1 million parameters. 
5. Examples of a longer output (if the bottom doesn't satisfy you) will be included in an additional folder. Notice that although it mimics Shakespeare's style well, a lot of the words and sentences it creates are nonsensical, which makes sense! Little Shakespeare's Size is only around a MB, which is a very small amount of data to be trained on. In order for us to have a better model which makes sense, we would need more examples and explanations of how Shakespeare wrote his works, along with more prompting. This is another reason why "big data" has become so prevalent across all of Machine Learning. We haven't learned how to initialize neural networks weights without a lot of data just yet. In nature, we have animals like horses who know how to walk and run almost immediately after birth. We have insects who know how to eat and grow and metamorphose and evolve from the second they become alive to the second they die. We just haven't learned how to exactly do that yet, which is why we need to learn that behavior. 

Improvements (TODO):
1. We can improve the tokenization of the words. Currently, our tokenization is based on character-level storage in the English Language, which means while our vocabulary size is low (only the letters in the alphabet along with punctuation), the amount of tokens that we need to generate and to train are enormous, horrendously slowing down our model. In order to fix this, we could use OpenAI's tokenization method, "tiktoken", which has a much larger vocabulary size of ~50000, but could drastically increase the speed of our model as well as make sure that it can be trained better.
2. There are a few ways we can train our model. Currently, our model is still slow because it the parallelization is not 100%. Since the device I am using is a MacBook Air, it lacks the parallelization that a GPU would have training a model. This can be solved by SSHing into something like AWS or Azure, or by using a PC with an NVIDIA chip. 
3. We currently can't prompt the model. Right now, all we are doing is generating text that is similar to Shakespeare, but we can't ask it certain questions like a ChatGPT. In order to do this, we can try to identify our question and raise it to the model as context, which will be encoded into our tokens and then decoded by the model. However, this probably needs some fine-tuning because our model doesn't currently have enough data.
4. Our validation loss is quite a bit higher than our training loss. Currently, with the parameters in "gpt.py", the training loss I received after 5000 steps was 1.23 compared to a validation loss of 1.35. In order to fix this, we could fine-tune the parameters like the dropout ratio (which is currently 0.2, though typically ranges from 0.2 - 0.5) or the batch and block size. Although a higher validation loss is expected (since our dataset is just too small), ideally, we should be able to lower it. 

Finally, here's an example of output I received:

    PAULINA:
    Ay, that I took it curase.

    MENENIUS:
    Gainster Rome; to me farewell country.

    Provost:
    Your honour gracemen, one silver goese that I
    answer honour honour you.

    ANGELO:
    And, sir, you that honours believe; yours,
    that puts us the token your virtue.

    LEONTES:
    Let her alms speak'd against her servant.

    First Murderer:
    Here's a Fortune tale part tastes;
    But to our hoped-fortuner estime hath
    Tooph-scarst empore us assisting aught their
    Full prodigy, like rouse bove gown of foul,
