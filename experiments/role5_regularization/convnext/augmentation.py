"""Shared CPU-side online augmentation; no model, training or GPU work here."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
from PIL import Image,ImageOps,ImageEnhance

HIGH={'cardigan','sweater','sock','scarf','watch','leg warmer','cape','jumpsuit','vest','umbrella','glove'}
GENERAL={'headband, head covering, hair accessory','skirt','coat'}
MID_WEIGHT={'sweater','sock','scarf','watch','headband, head covering, hair accessory','glove'}
POLICY={'version':1,'seed':42,'epochs':30,'draws_per_epoch':43197,'hflip_probability':0.5,
 'jitter':{'brightness':0.10,'contrast':0.10,'saturation':0.05,'hue':0,'high_probability':0.35,'general_probability':0.20},
 'sampling':{'leg warmer':2.5,'positive_count_below_1000':1.75,'middle_weak':1.30,'default':1.0,'combine':'maximum','cap':2.5},
 'extra_color_match':False,'crop':False,'rotation':False,'erasing':False,'mixup':False,'cutmix':False,
 'label_unknown':-1,'loss':'BCE over known entries only','validation_random_augmentation':False,
 'implementation_order':['EXIF','RGB','flip','jitter in stored order','aspect resize bicubic','center black pad','checkpoint normalize']}

def augmentation_probability(positive_classes):
 p=set(positive_classes)
 return 0.35 if p&HIGH else 0.20 if p&GENERAL else 0.0

def sampling_weight(positive_classes,counts):
 p=set(positive_classes)
 values=[1.0]
 if 'leg warmer' in p:values.append(2.5)
 if any(counts[c]<1000 for c in p):values.append(1.75)
 if p&MID_WEIGHT:values.append(1.30)
 return min(2.5,max(values))

def transform_rgb(image,flip=False,factors=(1.,1.,1.),order=(0,1,2)):
 im=ImageOps.exif_transpose(image).convert('RGB')
 if flip:im=ImageOps.mirror(im)
 classes=[ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color]
 for k in order:
  if factors[k]!=1.0:im=classes[int(k)](im).enhance(float(factors[k]))
 return im

def resize_pad(im,size):
 w,h=im.size;scale=size/max(w,h)
 im=im.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BICUBIC)
 canvas=Image.new('RGB',(size,size),(0,0,0))
 canvas.paste(im,((size-im.width)//2,(size-im.height)//2))
 return canvas

def tensor_array(image,size,mean,std,flip=False,factors=(1.,1.,1.),order=(0,1,2)):
 im=resize_pad(transform_rgb(image,flip,factors,order),size)
 arr=np.asarray(im,dtype=np.float32)/255
 return ((arr-np.asarray(mean,dtype=np.float32))/np.asarray(std,dtype=np.float32)).transpose(2,0,1).copy()

def make_epoch_plan(weights,jitter_prob,epoch,seed=42,draws=43197):
 rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,epoch])))
 weights=np.asarray(weights,dtype=float);prob=weights/weights.sum()
 ids=rng.choice(len(weights),size=draws,replace=True,p=prob).astype(np.int32)
 flip=rng.random(draws)<.5
 jitter=rng.random(draws)<np.asarray(jitter_prob)[ids]
 factors=np.ones((draws,3),dtype=np.float32)
 factors[jitter]=rng.uniform([.9,.9,.95],[1.1,1.1,1.05],size=(int(jitter.sum()),3))
 order=np.argsort(rng.random((draws,3)),axis=1).astype(np.uint8)
 return {'image_index':ids,'flip':flip,'jitter':jitter,'factors':factors,'order':order}

def masked_bce(logits,targets):
 import torch
 import torch.nn.functional as F
 known=targets>=0
 per=F.binary_cross_entropy_with_logits(logits.float(),targets.clamp(min=0),reduction='none')
 return (per*known).sum()/known.sum().clamp(min=1)

class ManifestDataset:
 """A map-style dataset whose index is a DRAW position, not an image index.

 Use DataLoader(shuffle=False), replace epoch_plan each epoch, and never add a
 second sampler/augmentation. Both models reuse exactly the same epoch files.
 For gradient accumulation, reduce summed known-label BCE over the full window's
 known-label count; sample-count weighting is insufficient with partial labels.
 """
 def __init__(self,manifest_path,epoch_plan,size,mean,std,roots):
  self.rows=list(csv.DictReader(Path(manifest_path).open(encoding='utf-8-sig',newline='')))
  self.plan=dict(np.load(epoch_plan,allow_pickle=False));self.size=size;self.mean=mean;self.std=std;self.roots=roots
 def __len__(self):return len(self.plan['image_index'])
 def __getitem__(self,k):
  import torch
  row=self.rows[int(self.plan['image_index'][k])]
  path=Path(self.roots[row['dataset']])/'images'/row['file_name']
  with Image.open(path) as im:
   x=tensor_array(im,self.size,self.mean,self.std,bool(self.plan['flip'][k]),self.plan['factors'][k],self.plan['order'][k])
  y=np.array(json.loads(row['targets']),dtype=np.float32)
  return torch.from_numpy(x),torch.from_numpy(y)
