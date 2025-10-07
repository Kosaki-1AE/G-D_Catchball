# -*- coding: utf-8 -*-
# Nori AV Shannon++ (All-in-one, Blue→Gray curiosity)
# - Video: OpticalFlow → Shannon
# - Audio: STFT → Shannon
# - Fuzzy fusion, Predictor, RewardSystem
# - CuriosityAgent: 青点("これは何?") → グレー点("そういうものかァ")
# - ChatAgent: persona切替, answerでグレー化
# - Backend Auto Detect: CUDA/OpenCL/CPU

import os
import queue
import re
import sys
import time
from collections import deque

import cv2 as cv
import numpy as np

# ========= Audio optional =========
USE_AUDIO = True
try:
    import sounddevice as sd
except Exception:
    USE_AUDIO = False
    sd = None
    print("[WARN] sounddevice が読み込めないので音声は無効化します。")

# ========= Config =========
CFG = dict(
    cam_id=0, frame_size=(640,360),
    flow_deadzone=0.03, flow_bins=12, flow_clip_percentile=99.0, flow_ema=0.25,
    sr=16000, blocksize=512, audio_bins=32, audio_ema=0.35,
    curiosity_dir="curiosity_snaps", curiosity_err_thresh=0.08,
    curiosity_hv_low=0.45, curiosity_hv_high=1.15,
    curiosity_cooldown=0.9, curiosity_sim_thresh=0.92, curiosity_topk=2,
    H_display_max=3.5,
)

# ========= Utils =========
def shannon_entropy_from_hist(h, eps=1e-12):
    s = float(np.sum(h))
    if s <= eps: return 0.0
    p = h / (s+eps); p = np.clip(p, eps, 1.0)
    return float(-np.sum(p*np.log(p)))

def ema(prev, x, a): return float(x) if prev is None else float(a*x+(1-a)*prev)
def fuzzy_low(value, low, high): return float(np.clip((high-value)/(high-low+1e-9),0,1))

def draw_meter(frame, value, min_v, max_v, x,y,w,h,label,color=(0,200,0)):
    v=(value-min_v)/(max_v-min_v+1e-9); v=float(np.clip(v,0,1))
    cv.rectangle(frame,(x,y),(x+w,y+h),(60,60,60),1)
    cv.rectangle(frame,(x,y),(x+int(w*v),y+h),color,-1)
    cv.putText(frame,f"{label}: {value:.3f}",(x,y-6),
               cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv.LINE_AA)

# ========= Predictor & Reward =========
class Predictor:
    def __init__(self, lam=0.98, eps=1e-6):
        self.lam,self.eps=lam,eps; self.params={}
    def _init(self,name,x0):
        self.params[name]={"a":0.9,"b":0.0,"P":np.eye(2)*10.0,"last":float(x0)}
    def update_and_predict(self,name,x):
        x=float(x)
        if name not in self.params: self._init(name,x); return x,0.0
        p=self.params[name]; x_prev=p["last"]
        phi=np.array([x_prev,1.0]); theta=np.array([p["a"],p["b"]])
        P=p["P"]/self.lam; denom=float(phi@P@phi)+self.eps
        K=(P@phi)/denom; err=x-float(phi@theta); theta=theta+K*err
        P=(np.eye(2)-np.outer(K,phi))@P
        p["a"],p["b"],p["P"],p["last"]=float(theta[0]),float(theta[1]),P,x
        return float(theta[0]*x+theta[1]),abs(float(err))

class RewardSystem:
    def __init__(self, enter=0.5, exit=0.6, dwell=45, wv=0.6, wa=0.4):
        self.enter,self.exit,self.dwell=float(enter),float(exit),int(dwell)
        self.wv,self.wa=float(wv),float(wa); self.alpha=0.2
        self.hist=deque(maxlen=60)
    def _smooth(self,attr,val,clamp):
        cur=getattr(self,attr); new=(1-self.alpha)*cur+self.alpha*val
        lo,hi=clamp
        if attr=="dwell": setattr(self,attr,int(np.clip(new,lo,hi)))
        else: setattr(self,attr,float(np.clip(new,lo,hi)))
    def step(self,Hv,Ha,S,ev,ea,es):
        Hvn=np.clip((Hv-0.35)/(1.10-0.35+1e-9),0,1)
        Han=np.clip((Ha-2.0)/(5.0-2.0+1e-9),0,1) if Ha is not None else 0.5
        evn,ean,esn=np.clip(ev/0.35,0,1),np.clip((ea or 0.0)/1.0,0,1),np.clip(es/0.20,0,1)
        good=(1-Hvn)*0.35+(1-Han)*0.15+(1-evn)*0.20+(1-ean)*0.10+(1-esn)*0.20
        self.hist.append(float(good))
        shock=0.4*evn+0.3*ean+0.3*esn; calm=(1-Hvn)*0.6+(1-evn)*0.4
        self._smooth("enter",self.enter+0.15*shock-0.1*calm,(0.35,0.75))
        self._smooth("exit",self.exit+0.10*shock,(0.45,0.85))
        self._smooth("dwell",self.dwell+30*shock-20*calm,(15,120))
        target_wa=np.clip(0.5-0.3*Han,0.1,0.6); target_wv=1.0-target_wa
        self._smooth("wa",target_wa,(0.1,0.8)); self._smooth("wv",target_wv,(0.2,0.9))
        return dict(enter=self.enter,exit=self.exit,dwell=int(self.dwell),
                    wv=self.wv,wa=self.wa,reward=float(good),
                    reward_avg=float(np.mean(self.hist)))

# ========= Curiosity (Blue→Gray) =========
class CuriosityAgent:
    def __init__(self,save_dir="curiosity_snaps",k=2,
                 err_thresh=0.08,hv_low=0.45,hv_high=1.15,
                 sim_thresh=0.92,cooldown=0.9):
        self.save_dir=save_dir; os.makedirs(save_dir,exist_ok=True)
        self.k=k; self.err_thresh=err_thresh
        self.hv_low,self.hv_high=hv_low,hv_high
        self.sim_thresh,self.cooldown=sim_thresh,cooldown
        self.last_feats=deque(maxlen=40); self.questions=deque(maxlen=30)
        self.last_fire_t=0.0; self.marks=deque(maxlen=120)
    def _push_mark(self,x,y,size,mode="what"):
        self.marks.append((int(x),int(y),int(size),time.time(),mode))
    def get_marks(self,decay=3.0):
        now=time.time()
        return [(x,y,size,mode) for (x,y,size,ts,mode) in self.marks if (now-ts)<decay]
    @staticmethod
    def _roi_feats(img):
        small=cv.resize(img,(32,32))
        hist=cv.calcHist([small],[0],None,[32],[0,256]).flatten()
        hist/= (np.sum(hist)+1e-9)
        return np.concatenate([small.flatten()/255.0,hist]).astype(np.float32)
    @staticmethod
    def _cos(a,b):
        na,nb=np.linalg.norm(a),np.linalg.norm(b)
        if na<1e-9 or nb<1e-9: return 0.0
        return float(np.dot(a,b)/(na*nb))
    def _novel(self,feat):
        if not self.last_feats: return True
        return (max(self._cos(feat,f) for f in self.last_feats)<self.sim_thresh)
    def step(self,frame_bgr,gray,flow,H_video_ema,e_video):
        now=time.time()
        if not(self.hv_low<=(H_video_ema or 0.0)<=self.hv_high): return None
        if e_video<self.err_thresh: return None
        if (now-self.last_fire_t)<self.cooldown: return None
        fx,fy=flow[...,0],flow[...,1]; mag=np.sqrt(fx*fx+fy*fy)
        H,W=mag.shape; step=32; cands=[]
        for y in range(0,H-step,step):
            for x in range(0,W-step,step):
                tile=mag[y:y+step,x:x+step]; v=float(np.max(tile))
                yy,xx=np.unravel_index(np.argmax(tile),tile.shape)
                cands.append((v,x+xx,y+yy,step))
        if not cands: return None
        cands.sort(reverse=True); picks=cands[:self.k]
        for _,cx0,cy0,sz in picks: self._push_mark(cx0,cy0,sz,"what")
        for _,cx,cy,sz in picks:
            x0=max(0,cx-64); x1=min(W,cx+64); y0=max(0,cy-64); y1=min(H,cy+64)
            roi=gray[y0:y1,x0:x1]
            if roi.size<64*64: continue
            feat=self._roi_feats(roi)
            if not self._novel(feat): continue
            self._push_mark(cx,cy,sz,"what")
            path=f"{self.save_dir}/roi_{int(time.time())}_{cx}x{cy}.png"
            cv.imwrite(path,frame_bgr[y0:y1,x0:x1])
            self.last_feats.append(feat)
            self.questions.append((path,(x0,y0,x1,y1),now,f"これは何？@({cx},{cy})"))
            self.last_fire_t=now; return f"これは何？@({cx},{cy})"
        return None
    def latest_question(self): return self.questions[-1] if self.questions else None
    def answer_latest(self,ans_text):
        if not self.questions: return "（質問なし）"
        path,bbox,ts,qtext=self.questions[-1]
        base,ext=path.rsplit(".",1); new_path=f"{base}__{ans_text}.{ext}"
        os.rename(path,new_path)
        self.questions[-1]=(new_path,bbox,ts,qtext)
        if self.marks: x,y,size,t,mode=self.marks[-1]; self.marks[-1]=(x,y,size,t,"accepted")
        return f"そういうものかァ → {new_path}"

# ========= ChatAgent =========
class ChatAgent:
    def __init__(self,default_persona="univ_girl"):
        self.persona=default_persona
        self.last_reply=self._style("準備OK〜 'help'でコマンド見れるよ")
    def _style(self,text):
        if self.persona=="univ_girl": return f"{text} って感じかな〜"
        if self.persona=="polite": return f"{text}。よろしくお願いいたします。"
        if self.persona=="deadpan": return f"{text}。"
        return text
    def reply(self,text,ctx):
        if "answer=" in text: return self._style(ctx['curiosity'].answer_latest(text.split("=",1)[1].strip()))
        if text.lower()=="what": 
            q=ctx['curiosity'].latest_question(); return self._style(q[3] if q else "まだ質問ないかも〜")
        return self._style("お題ちょーだい")

# ========= Audio =========
audio_q=queue.Queue()
def audio_callback(indata,frames,time_info,status): audio_q.put(indata.copy())
def start_audio():
    if not USE_AUDIO: return
    try:
        stream=sd.InputStream(channels=1,samplerate=CFG["sr"],
                              blocksize=CFG["blocksize"],callback=audio_callback)
        stream.start(); return stream
    except Exception as e:
        print("[WARN] audio失敗:",e); return None
def pop_latest_audio_block(q):
    try:
        while True: latest=q.get_nowait()
    except queue.Empty: return locals().get("latest",None)

def audio_shannon(block,bins=32):
    if block is None: return 0.0
    x=block[:,0]; win=np.hanning(len(x)); X=np.abs(np.fft.rfft(x*win))**2
    hi=np.percentile(X,99)+1e-6; h,_=np.histogram(X,bins=bins,range=(0,hi))
    return shannon_entropy_from_hist(h)

# ========= Main Loop =========
cap=cv.VideoCapture(CFG["cam_id"]); assert cap.isOpened()
curiosity=CuriosityAgent(); chat=ChatAgent()
pred=Predictor(); rew=RewardSystem()
audio_stream=start_audio()

H_video_ema=None; H_audio_ema=None; state="MOTION"; still_counter=0
last_chat_text=chat.last_reply; last_chat_time=0

while True:
    ok,frame=cap.read(); 
    if not ok: break
    frame=cv.resize(frame,CFG["frame_size"]); gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    # Flow
    if 'prev_gray' not in locals(): prev_gray=gray; continue
    flow=cv.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
    fx,fy=flow[...,0],flow[...,1]; mag=np.sqrt(fx*fx+fy*fy)
    mag_max=np.percentile(mag,CFG["flow_clip_percentile"])+1e-6
    h_flow,_=np.histogram(np.clip(mag,0,mag_max),bins=CFG["flow_bins"],range=(0,mag_max))
    H_video=shannon_entropy_from_hist(h_flow); H_video_ema=ema(H_video_ema,H_video,CFG["flow_ema"])

    # Audio
    H_audio=None; block=pop_latest_audio_block(audio_q)
    if block is not None: H_audio=audio_shannon(block,CFG["audio_bins"]); H_audio_ema=ema(H_audio_ema,H_audio,CFG["audio_ema"])

    # Curiosity
    qtext=curiosity.step(frame,gray,flow,H_video_ema or 0.0,0.1)
    if qtext: print("[Q]",qtext)

    # Draw marks
    for x,y,size,mode in curiosity.get_marks():
        if mode=="what": color=(255,0,0); label="?"
        elif mode=="accepted": color=(128,128,128); label="…"
        else: color=(200,200,200); label=""
        cv.circle(frame,(x,y),max(3,size//10),color,-1)
        if label: cv.putText(frame,label,(x+6,y-6),cv.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # Chat input key
    key=cv.waitKey(1)&0xFF
    if key in (27,ord('q')): break
    elif key==ord('c'):
        user=input("[CHAT]> ")
        ctx=dict(curiosity=curiosity)
        last_chat_text=chat.reply(user,ctx); last_chat_time=time.time()

    if time.time()-last_chat_time<5:
        cv.putText(frame,f"BOT: {last_chat_text}",(20,330),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

    cv.imshow("Nori AV Shannon++",frame); prev_gray=gray

cap.release(); cv.destroyAllWindows()
