from typing import List, Union
import torch
import clip
import PIL
from PIL import Image
import requests
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

class ZeroShotImageIdentification():

  def __init__(self, 
               *args, 
               **kwargs):
   
         if "lang" in kwargs:
            self.lang = kwargs["lang"]
         else:
            self.lang = "en"

         lang_codes = self.available_languages()

         if self.lang not in lang_codes:
            raise Exception('Language code {} not valid, supported codes are {} '.format(self.lang, lang_codes))
            return 

         device = "cuda:0" if torch.cuda.is_available() else "cpu" 

         if self.lang == "en":
            model_tag = "ViT-B/32"
            if "model" in kwargs:
                model_tag = kwargs["model"] 
            print("Loading OpenAI CLIP model {} ...".format(model_tag))    
            self.model, self.preprocess = clip.load(model_tag, device=device) 
            print("Label language {} ...".format(self.lang))
         else:          
            model_tag = "clip-ViT-B-32"
            print("Loading sentence transformer model {} ...".format(model_tag))
            self.model = SentenceTransformer('clip-ViT-B-32', device=device)
            self.text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device=device)
            print("Label language {} ...".format(self.lang))

  def available_models(self):
      return clip.available_models()

  def available_languages(self):
      codes = """ar, bg, ca, cs, da, de, en, el, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, 
      hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt, pt-br, 
      ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw"""
      return set([code.strip() for code in codes.split(",")])

  def _load_image(self, image: str) -> "PIL.Image.Image":
    
      if isinstance(image, str):
          if image.startswith("http://") or image.startswith("https://"):
              image = PIL.Image.open(requests.get(image, stream=True).raw)
          elif os.path.isfile(image):
              image = PIL.Image.open(image)
          else:
              raise ValueError(
                  f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
              )
      elif isinstance(image, PIL.Image.Image):
          image = image
      else:
          raise ValueError(
              "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
          )
      image = PIL.ImageOps.exif_transpose(image)
      image = image.convert("RGB")
      return image            

  def __call__(
        self, 
        image: str,
        candidate_labels: Union[str, List[str]],
        *args,
        **kwargs,
    ):
    
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if self.lang == "en":
            if "hypothesis_template" in kwargs:
                hypothesis_template = kwargs["hypothesis_template"] 
            else:
                hypothesis_template = "A photo of {}"

            if isinstance(candidate_labels, str):
              labels = [hypothesis_template.format(candidate_label) for candidate_label in candidate_labels.split(",")]
            else:    
              labels = [hypothesis_template.format(candidate_label) for candidate_label in candidate_labels]
        else:
            if "hypothesis_template" in kwargs:
                hypothesis_template = kwargs["hypothesis_template"] 
            else:
                hypothesis_template = "{}"

            if isinstance(candidate_labels, str):
              labels = [hypothesis_template.format(candidate_label) for candidate_label in candidate_labels.split(",")]
            else:    
              labels = [hypothesis_template.format(candidate_label) for candidate_label in candidate_labels]
        if  "top_k" in kwargs:
             top_k = kwargs["top_k"] 
        else:
             top_k = len(labels)
        
        if str(type(self.model)) == "<class 'clip.model.CLIP'>":
            img = self.preprocess(self._load_image(image)).unsqueeze(0).to(device)
            text = clip.tokenize(labels).to(device)
            image_features = self.model.encode_image(img)
            text_features = self.model.encode_text(text)
        else:    
            image_features = torch.tensor(self.model.encode(self._load_image(image)))
            text_features = torch.tensor(self.text_model.encode(labels))
        
        sim_scores = util.cos_sim(text_features, image_features)
        out = []
        for sim_score in sim_scores:
            out.append(sim_score.item() * 100)
        probs = torch.tensor([out])
        probs = probs.softmax(dim=-1).cpu().numpy()
        scores = list(probs.flatten())
        
        sorted_sl = sorted(zip(scores, candidate_labels), key=lambda t:t[0], reverse=True)  
        scores, candidate_labels = zip(*sorted_sl)
        
        preds = {}
        preds["image"] = image
        preds["scores"] = scores
        preds["labels"] = candidate_labels
        return preds
      
