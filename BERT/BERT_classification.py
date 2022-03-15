from posixpath import split
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from nlp import load_dataset

import torch
import numpy as np

dataset = load_dataset('csv', data_files='./imdbs.csv', split='train')
# print(type(dataset))

dataset = dataset.train_test_split(test_size= 0.3)
# print(dataset)

train_set = dataset['train']
test_set = dataset['test']

model =BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# sentence = '''One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me. The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word. It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away. I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side.'''
# print(sentence)
# print(tokenizer.tokenize(sentence))
# print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))
# print('='*50)
# print(tokenizer(sentence))

# DATA preprocessing
def preprocess(data):
    return tokenizer(data['text'], padding= True, truncation= True)

train_set = train_set.map(preprocess, batched= True, batch_size= len(train_set))
test_set = test_set.map(preprocess, batched= True, batch_size= len(test_set))

train_set.set_format('torch', columns= ['input_ids', 'attention_mask', 'label'])
test_set.set_format('torch', columns= ['input_ids', 'attention_mask', 'label'])

# Model Training
batch_size = 8
epochs = 10
warmup_steps = 500
weight_decay = 0.01

training_arg = TrainingArguments(
    output_dir= './results',
    num_train_epochs= epochs,
    per_device_train_batch_size= batch_size,
    per_device_eval_batch_size= batch_size,
    warmup_steps= warmup_steps,
    weight_decay= weight_decay,
    # evaluate_during_training= True,
    logging_dir= './logs'
)

trainer = Trainer(
    model = model,
    args= training_arg,
    train_dataset= train_set,
    eval_dataset= test_set
)

trainer.train()
trainer.evaluate()