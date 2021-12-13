# Databricks notebook source
# MAGIC %pip install deepspeech
# MAGIC %pip install torchaudio
# MAGIC %pip install omegaconf
# MAGIC %pip install transformers
# MAGIC %pip install textblob

# COMMAND ----------

import os, wave
from IPython.display import Audio
import numpy as np
from deepspeech import Model
import torch
import zipfile
import torchaudio
import mlflow
from glob import glob
from transformers import pipeline

# COMMAND ----------

os.system('curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm') # language model
os.system('curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer') # acoustic model
os.system ('curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/audio-0.9.3.tar.gz')
os.system('tar xvf audio-0.9.3.tar.gz')

# COMMAND ----------

Audio('audio/2830-3980-0043.wav')

# COMMAND ----------

# MAGIC %sh ls audio/*.wav

# COMMAND ----------

# MAGIC %sh deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio audio/2830-3980-0043.wav

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we can quickly do the same with python as well

# COMMAND ----------

model = Model('deepspeech-0.9.3-models.pbmm') 
model.enableExternalScorer('deepspeech-0.9.3-models.scorer')
def read_wav_file(filename):
  """ read the wav file and get the frames 
  and push the frames to buffer"""
  with wave.open(filename, 'rb') as w:
    rate = w.getframerate()
    frames = w.getnframes()
    print(f'INFO: {filename} frame rate is {rate}')
    buffer = w.readframes(frames)
  return buffer,rate

# COMMAND ----------

def transcribe_batch(audio_file):
  """score a batch"""
  buffer, rate = read_wav_file(audio_file)
  data16 = np.frombuffer(buffer, dtype=np.int16)
  return model.stt(data16)
  

# COMMAND ----------

transcribe_batch('audio/2830-3980-0043.wav')

# COMMAND ----------

# MAGIC %md Try with custom recording

# COMMAND ----------

dbfs_path='/Users/sathish.gangichetty@databricks.com/stt/data/'
dbutils.fs.mkdirs(dbfs_path)
# dbutils.fs.mv('dbfs:/FileStore/sg_audio.wav',dbfs_path)
# dbutils.fs.mv('dbfs:/FileStore/sg_audio2.wav',dbfs_path)

# COMMAND ----------

dbutils.fs.ls(dbfs_path)

# COMMAND ----------

custom_audio_recording = f'/dbfs/{dbfs_path}/sg_audio.wav'
Audio(custom_audio_recording)

# COMMAND ----------

transcribe_batch(custom_audio_recording)

# COMMAND ----------

# MAGIC %md More real life like stuff. with stops and stutters.

# COMMAND ----------

# dbutils.fs.mv('dbfs:/FileStore/sg_navy.wav',dbfs_path)
dbfs_path='/Users/sathish.gangichetty@databricks.com/stt/data/'
custom_real_audio_recording = f'/dbfs/{dbfs_path}/sg_navy.wav'
Audio(custom_real_audio_recording)

# COMMAND ----------

transcribe_batch(custom_real_audio_recording)

# COMMAND ----------

# from scipy.io import wavfile
# import noisereduce as nr
# # load data
# rate, data = wavfile.read(custom_real_audio_recording)
# # perform noise reduction
# reduced_noise = nr.reduce_noise(y=data, sr=rate)
# data16 = np.frombuffer(reduced_noise, dtype=np.int16)
# model.stt(data16)

# COMMAND ----------

# dbutils.fs.mv('dbfs:/FileStore/sg_navy2.wav',dbfs_path)

# COMMAND ----------

custom_navyfaq_recording = f'/dbfs/{dbfs_path}/sg_navy2.wav'
Audio(custom_navyfaq_recording)

# COMMAND ----------

transcribe_batch(custom_navyfaq_recording)

# COMMAND ----------

import torch
import zipfile
import torchaudio
from glob import glob

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

# download a single file, any format compatible with TorchAudio
torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
                               dst ='speech_orig.wav', progress=True)
test_files = glob('speech_orig.wav')
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

output = model(input)
for example in output:
    print(decoder(example.cpu()))

# COMMAND ----------

Audio('speech_orig.wav')

# COMMAND ----------

# MAGIC %sh ls /root/.cache/torch/hub/snakers4_silero-models_master

# COMMAND ----------

def _process_file(file):
  device = torch.device('cpu')
  batches = split_into_batches(glob(file), batch_size=10)
  input = prepare_model_input(read_batch(batches[0]),
                              device=device)

  output = model(input)
  for example in output:
      return decoder(example.cpu())

# COMMAND ----------

Audio(custom_audio_recording)

# COMMAND ----------

result = _process_file(custom_audio_recording)
result

# COMMAND ----------

from textblob import TextBlob
result_text = TextBlob(result)
result_corrected = result_text.correct()
result_corrected

# COMMAND ----------

classifier = pipeline('sentiment-analysis')
classifier(result)

# COMMAND ----------

# MAGIC %md 
# MAGIC if you want to save model to disk and finetune later you might want to do this

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# COMMAND ----------

tokens =tokenizer(result, padding =True, truncation=True,return_tensors='pt')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# COMMAND ----------

# print(model.config.id2label)
if (torch.argmax(torch.nn.functional.softmax(model(tokens['input_ids']).logits,dim=1)[0]) == 0):
  print(f'INFO: Result is : {model.config.id2label[0]}')
else:
  print(f'INFO:Result is : {model.config.id2label[1]}')

# COMMAND ----------

with mlflow.start_run(run_name='model_logging'):
  mlflow.pytorch.log_model(model,checkpoint)

# COMMAND ----------

lmodel = mlflow.pytorch.load_model("dbfs:/databricks/mlflow-tracking/2906506356160478/a6bd16249dbf4e79bce0a77cc43887a6/artifacts/distilbert-base-uncased-finetuned-sst-2-english")

# COMMAND ----------

lmodel(tokens['input_ids'])

# COMMAND ----------

model(tokens['input_ids'])
