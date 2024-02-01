# DE-COP
This is the anonymized version of the code and data for the paper 'DE-COP: Detecting Copyrighted Content in Language Models Training Data' to be submitted to ICML 2024.<br>


DE-COP is a method for Detecting Copyrighted Content in the Language Models Training Data. It employs probing tasks structured as multiple-choice questions, whose options include both verbatim text and their paraphrases.
![DE-COP](https://github.com/avduarte333/DE-COP/assets/79573601/78b2f167-988a-48cf-aef7-805682508873)


---
## DE-COP Example
⚠ Important: When using API-based models, ensure to add the API key in 2_eval_BlackBox.py<br>

First, oversample each document according to all the possible 4-Option Permutations.<br>
A new .xlsx file with the results will be created for each document. 
```
cd test_example
python 1_oversample_labels.py <file_with_document_names.txt>
```
DE-COP Evaluation:
- If Model is ChatGPT or Claude
```
python 2_eval_BlackBox.py <file_with_document_names.txt> <black_box_model_name>

#In example:
python 2_eval_BlackBox.py 0_book_list.txt ChatGPT
```

- If Model is ChatGPT or Claude
```
python 2_eval_HF.py <file_with_document_names.txt> <hf_model_name>

#In example:
python 2_eval_HF.py 0_book_list.txt LLaMA-2-70B
```


### 📚 arXivTection and BookTection Datasets
The arXivTection and the BookTection datasets serve as benchmarks designed for the task of detecting pretraining data from Large Language models.

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

### 🧪 Testing Models on the Benchmarks
Our datasets are planned to be used on a Multiple-Choice-Question-Answering format. Nonetheless, it is compatible to be used with other pretraining data detection methods.<br>

<br>
<br>

### 🤝 Compatibility
The Multiple-Choice-Question-Answering task with our Dataset is designed to be applied to various models, such as:<br>
- LLaMA-2
- Mistral
- Mixtral
- Chat-GPT (gpt-3.5-turbo-instruct)
- GPT-3 (text-davinci-003)
- Claude 
