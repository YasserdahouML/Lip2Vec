"""
eval functions used in main.py
"""
import torch
from transformers import Wav2Vec2CTCTokenizer
from jiwer import wer
import util.misc as utils
from models.wav2vec2 import Wav2Vec2
from metrics.measures import get_wer as wer_measure


@torch.no_grad()
def test_wer(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLoggerMulti(delimiter="  ")
    header = 'Testing'
    wav2vec = Wav2Vec2().to(device).eval()
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

    for video, label in metric_logger.log_every(data_loader, 10, header):
        video = video.to(device)
        
        outputs = model.module.forward_test(video)
        
        preds = wav2vec(outputs)
        preds = torch.argmax(preds, dim=-1)
        transcription = tokenizer.batch_decode(preds)

        for i in range(0,len(outputs)):
            
            print(f"Lip2Vec: {transcription[i]}")
                    
            print(f"Ground-truth: {label[i]}")
            
            wer_score = wer_measure(transcription[i], label[i])
            
            metric_logger.updateWer(wer_score*100, len(label[i].split()))

            
    wer_score = metric_logger.synchronize_between_processes(eval=True)
    
    if wer_score is not None:
        print("Averaged WER is:", wer_score[2].cpu().item())
    return wer_score