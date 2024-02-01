# DE-COP
This is the anonymized version of the code and data for the paper 'DE-COP: Detecting Copyrighted Content in Language Models Training Data' to be submitted to ICML 2024


# 📄 arXivTection and BookTection Datasets
The arXivTection and the BookTection datasets serve as a benchmarks designed for the task of detecting pretraining data from Large Language models.

The arXivTection consists of 50 research papers extracted from arXiv. 
- 25 published in 2023: Non-Training data, "_label_" column = 0.
- 25 published before 2022: Training data, "_label_" column = 1.

The BookTection consists of 165 books. 
- 60 published in 2023: Non-Training data, "_label_" column = 0.
- 105 published before 2022: Training data, "_label_" column = 1.


From each paper / book ≈ 30 passages are extracted. Each passage is paraphrased 3 times using the Language Model Claude v2.0. <br>
The "_Answer_" column indicates which of the passages is the real excerpt.<br>
Passages on arXivTection are extracted to be on average ≈ 128 tokens in length.<br>
Passages on BookTection come in 3 different sizes (small, medium and large) which aim to be respectively ≈(64, 128 and 256) tokens in length.

<br>
<br>

# 🧪 Testing Models on the Benchmarks
Our datasets are planned to be used on a Multiple-Choice-Question-Answering format. Nonetheless, it is compatible to be used with other pretraining data detection methods.<br>

<br>
<br>

# 🤝 Compatibility
The Multiple-Choice-Question-Answering task with our Dataset is designed to be applied to various models, such as:<br>
- LLaMA-2
- Mistral
- Mixtral
- Chat-GPT (gpt-3.5-turbo-instruct)
- GPT-3 (text-davinci-003)
- Claude 

<br>
<br>

# 🔧 Loading the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("avduarte333/arXivTection")
```
