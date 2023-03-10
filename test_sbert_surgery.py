###Test sbert
from sentence_transformers import SentenceTransformer
import sentence_transformers
import torch
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences = ['This framework generates embeddings for each input sentence','who are you?']

tokenised_sentences = sbert_model.tokenize(sentences)

tokenised_sentences = sentence_transformers.util.batch_to_device(tokenised_sentences,torch.device('cuda'))

print(sbert_model.encode(sentences))


####Now try using pytorch to get the embeddings from thr sbert model
sbert_model.to(torch.device('cuda'))
sbert_forward = sbert_model.forward(tokenised_sentences)

print(sbert_forward["sentence_embedding"])

print(sbert_forward["input_ids"])

print(sbert_forward.keys())

##Print shapes of all values
for key in sbert_forward.keys():
    print(key, sbert_forward[key].shape)

