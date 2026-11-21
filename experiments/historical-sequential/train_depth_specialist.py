import os,json,time,argparse,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration,AutoProcessor,BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
R=Path('/root/autodl-tmp/WISE-Read/open3dvqa_clean_rebuild_v1'); MP='/root/autodl-tmp/models/Qwen2-VL-7B-Instruct'; ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=10000);args=ap.parse_args(); rows=[json.loads(x) for x in open(R/'full_depth_sprint_v1/training/large_train_manifest.jsonl')][:args.limit];cache=R/'full_depth_sprint_v1/depth/cache';cache.mkdir(parents=True,exist_ok=True)
def depth_img(path):
 out=cache/(hashlib.sha1(path.encode()).hexdigest()+'.png')
 if not out.exists():
  a=np.nan_to_num(np.load(path).astype('float32')); pos=a[a>0];lo,hi=np.percentile(pos,[2,98]) if pos.size else (0,1);z=np.clip((a-lo)/max(hi-lo,1e-6),0,1);Image.fromarray((255*(1-z)).astype('uint8')).convert('RGB').resize((416,416)).save(out)
 return str(out)
q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True);model=Qwen2VLForConditionalGeneration.from_pretrained(MP,quantization_config=q,torch_dtype=torch.float16,device_map='auto',local_files_only=True);proc=AutoProcessor.from_pretrained(MP,local_files_only=True,min_pixels=256*28*28,max_pixels=576*28*28);model.config.use_cache=False;model=prepare_model_for_kbit_training(model);model.gradient_checkpointing_enable();model.enable_input_require_grads();model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'));opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4);out=R/'full_depth_sprint_v1/checkpoints/q2_depth_specialist';out.mkdir(parents=True,exist_ok=True);log=R/'full_depth_sprint_v1/depth/training_log.jsonl';log.write_text('');start=time.time()
for step,r in enumerate(rows,1):
 dp=depth_img(r['depth_path']);messages=[{'role':'user','content':[{'type':'image','image':dp},{'type':'text','text':r['question']+'\nAnswer concisely.'}]},{'role':'assistant','content':[{'type':'text','text':r['ground_truth']}]}];text=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=False);ims,vid=process_vision_info(messages);inp=proc(text=[text],images=ims,videos=vid,padding=True,return_tensors='pt');inp={k:(v.to(model.device) if hasattr(v,'to') else v) for k,v in inp.items()};labels=inp['input_ids'].clone();labels[labels==proc.tokenizer.pad_token_id]=-100;prefix=proc(text=[proc.apply_chat_template(messages[:1],tokenize=False,add_generation_prompt=True)],images=ims,videos=vid,padding=True,return_tensors='pt')['input_ids'].shape[1];labels[:,:prefix]=-100;inp['labels']=labels;torch.cuda.reset_peak_memory_stats();t=time.time();o=model(**inp);loss=o.loss;loss.backward();grad=(sum(p.grad.detach().float().norm().item()**2 for p in model.parameters() if p.grad is not None))**.5;opt.step();opt.zero_grad(set_to_none=True);rec={'step':step,'sample_id':r['sample_id'],'loss':float(loss.detach()),'step_time_s':time.time()-t,'peak_vram_mb':torch.cuda.max_memory_allocated()/1024/1024};
 with log.open('a') as f:f.write(json.dumps(rec)+'\n')
 if step in (5000,10000):model.save_pretrained(out/f'step_{step:05d}')
 if step%100==0:print('PROGRESS',step,'/',len(rows),'elapsed',round(time.time()-start,1),'sec_step',round((time.time()-start)/step,3),flush=True)
model.save_pretrained(out/'final');(out/'TRAINING_STATUS.json').write_text(json.dumps({'status':'COMPLETE','steps':len(rows),'elapsed_s':time.time()-start,'checkpoint':str(out/'final'),'representation':'official process_depth.py percentile 2/98 inverse normalized RGB 416x416','qa_image_disjoint':True,'scene_disjoint':False},indent=2));print('DONE',len(rows),flush=True)
