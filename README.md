A continuation of the backpropagation and makemore project, GPT is trying to emulate the Chat-GPT model from OpenAI in a much smaller and not as time-sensitive manner. In order to do this, let's try to create a character tokenization method that is similar to ChatGPT's tokenization (but smaller!), and make a simpler but just as powerful version as OpenAI's ChatGPT-2 model! 

In order to do this, let's analyze some papers and some important realizations that help us to make this model:
    1) First, a baseline setup is inspired by the paper <a href = "https://arxiv.org/pdf/1706.03762" target = "_blank">"Attention is All You Need"</a>, which emphasizes the need of self-attention and parallel attention in order to create blocks in which values, keys, and queries can talk to each other to determine values like the consonants, vowels, and words that we use in LLMs. We also added position-wise feed-forward networks and positional encoding in order to make sure the model is smarter and can generate better results.

Introduction:
GPTs are generally transformers that are based on "in-context" learning, such that they are mainly trained one time on massive scales of data in order to be good at essentially everything. For example, GPT-3 was trained such that it developed over 175 billion parameters, and this is contrasted with other fine-tuned models, which have specific datasets specific to desired tasks. GPTs are not inherently fine tuned, because it is expensive and requires new large datasets for every task, making it non task-agnostic. GPTs are generally designed to be able to answer questons based on a broad spectrum of topics, but their categories are separated as such:
    1) Zero-shot: The model predicts the answer given only a natural language description of the task.
    2) One-shot: In additon to the task description, the model sees a single xample of the task
    3) Few-shot: In addition to the task description, the model sees a few examples of the task. 

This approach allows for "maximum convenience, potential for robustness, and avoidance of spurious correlations, but is also the most challenging setting". This is because the ambiguousness of some tasks can even confuse humans, and it can be unclear as to what the user is actually searching for (at least in reference to the zero-shot approach above). 

Usage:
    1) Based on the desired use-case, this model can perform up to GPT-2 standards. However, this takes a very long time of running, even with only up to 5000 steps. In order to train the baseline model that high, you would need approximately 3-4 days in total with CUDA to perform that well. If you are using something like a MacBook (which lacks a NVIDIA GPU), then instead this will take significantly longer. Adjust the model parameters (at the top of "gpt_final_version.py") to make the batch size, block_size, embedding parameters, head and layer parameters smaller so that there will be a faster result. 
    2) Additionally, there are extra imports at the top of the page. All the PyTorch modules are self-explanatory, but we use the dataclass module in order to set-up a model configuration that would work for the the rest of the classes. 
    3) With the way the current parameters are set-up, I got my model to train through 5000 steps in approximately ~ 3 hours and 15 minutes
    4) In order to save the model, define a path to where you want to save the model. Then, you can use torch.save(model.state_dict(), model_save_path) in order to save it. In order to load it, you can define a normal model again from the same class (i.e. model = GPT(config)), then you can use model.load_state_dict(torch.load(model_save_path)) in order to load the saved state dictionary into the model, and set the model into evaluation mode (model.eval())

Finally, here's an example of output I received:
    * PAULINA:
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