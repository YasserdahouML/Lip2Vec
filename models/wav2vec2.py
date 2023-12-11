import torch.nn as nn
from transformers import Wav2Vec2CTCTokenizer
from .wav2vec2_util import Wav2Vec2ForCTC

class Wav2Vec2(nn.Module):

    def __init__(self):
        super().__init__()
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.config = model.config
        self.encoder = model.wav2vec2.encoder
        self.dropout = model.dropout
        self.head = model.lm_head
        self.model = model


    def forward(self, feat, atn_mask=None):
        """
        feat: input size [B] x T x 1024
        """
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        assert feat.shape[2] == 1024, 'Feature with size [Batch] x Tlen x 1024 to be input to wav2vec2 encoder'

        x = self.encoder(feat, attention_mask=atn_mask)['last_hidden_state']
        x = self.dropout(x)
        x = self.head(x)

        return x
    
    def forward_audio(self, audio):

        x = self.model(audio)[0]
        print(x.shape)
        return x


    def encode_feat(self, feat, atn_mask=None):


        x = self.encoder(feat, attention_mask=atn_mask)['last_hidden_state']
        x = self.dropout(x)
        return x


    def init_tokenizer(self):
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
