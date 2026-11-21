import json,time,hashlib,torch
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration,AutoProcessor,BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import PeftModel
R=Path('/root/autodl-tmp/WISE-Read/open3dvqa_clean_rebuild_v1');MP='/root/autodl-tmp/models/Qwen2-VL-7B-Instruct';RGB=R/'full_depth_sprint_v1/checkpoints/q2_large/final';DEP=R/'full_depth_sprint_v1/checkpoints/q2_depth_specialist/final';out=R/'full_depth_sprint_v1/depth/q2_rgb_depth_predictions.jsonl';dev=[json.loads(x) for x in open(R/'dataset/dev_clean_300_manifest.jsonl')]
cache=R/'full_depth_sprint_v1/depth/dev_cache';cache.mkdir(parents=True,exist_ok=True)
def dimg(r):
 p=cache/(r['sample_uid']+'.png')
 if not p.exists():
  a=np.nan_to_num(np.load(r['depth']['path']).astype('float32'));pos=a[a>0];lo,hi=np.percentile(pos,[2,98]) if pos.size else (0,1);z=np.clip((a-lo)/max(hi-lo,1e-6),0,1);Image.fromarray((255*(1-z)).astype('uint8')).convert('RGB').resize((416,416)).save(p)
 return str(p)
q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True);base=Qwen2VLForConditionalGeneration.from_pretrained(MP,quantization_config=q,torch_dtype=torch.float16,device_map='auto',local_files_only=True);proc=AutoProcessor.from_pretrained(MP,local_files_only=True,min_pixels=256*28*28,max_pixels=576*28*28);rgb=PeftModel.from_pretrained(base,str(RGB),is_trainable=False,adapter_name='rgb').eval();dep=PeftModel.from_pretrained(base,str(DEP),is_trainable=False,adapter_name='depth').eval()
def call(model,img,text):
 msg=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':text}]}];tx=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True);ims,vid=process_vision_info(msg);inp=proc(text=[tx],images=ims,videos=vid,padding=True,return_tensors='pt').to(model.device);torch.cuda.reset_peak_memory_stats();t=time.time();
 with torch.inference_mode():o=model.generate(**inp,max_new_tokens=64,do_sample=False,temperature=0.01,top_p=0.001,top_k=1)
 s=proc.batch_decode([z[len(i):] for i,z in zip(inp.input_ids,o)],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip();return s,time.time()-t,torch.cuda.max_memory_allocated()/1024/1024
with out.open('w') as f:
 for i,r in enumerate(dev,1):
  init,la,ma=call(rgb,r['rgb']['path'],r['question']+'\nAnswer using RGB and provide an initial reasoning state, then answer concisely.')
  ev,ld,md=call(dep,dimg(r),r['question']+'\nProvide concise complementary depth evidence relevant to the question. Do not answer beyond the depth image.')
  final,lf,mf=call(rgb,r['rgb']['path'],r['question']+'\nCombine the initial RGB state and complementary depth evidence. Provide only the final answer concisely.\nRGB state:\n'+init+'\nDepth evidence:\n'+ev)
  f.write(json.dumps({'sample_uid':r['sample_uid'],'question':r['question'],'reference_answer':r['reference_answer'],'rgb_initial':init,'depth_evidence':ev,'prediction':final,'latency_s':la+ld+lf,'peak_vram_mb':max(ma,md,mf),'status':'ok'},ensure_ascii=False)+'\n');f.flush()
  if i%25==0:print('PROGRESS',i,flush=True)
print('DONE',len(dev),flush=True)
