#!/usr/bin/env python3
\"\"\"
AutoScanlate AR — مترجم كوميكس/مانغا تلقائي
يدعم: صور منفردة, ZIP, CBZ, CBR
الاستخدام:
  python autoscanlate.py --input chapter.cbz --api_key sk-ant-...
  python autoscanlate.py --input pages/ --api_key sk-ant-...
\"\"\"

import os, sys, json, base64, zipfile, shutil, time, argparse, re
import urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SUPPORTED_IMG  = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    'C:/Windows/Fonts/arial.ttf',
    '/System/Library/Fonts/Helvetica.ttc',
]
CLAUDE_MODEL  = 'claude-sonnet-4-20250514'
MAX_TOKENS    = 4096
MAX_WORKERS   = 4
JPEG_QUALITY  = 92

def get_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, max(8, int(size)))
            except: pass
    return ImageFont.load_default()

def wrap_text(draw, text, font, max_w):
    words = text.split(); lines, cur = [], ''
    for w in words:
        test = (cur + ' ' + w).strip()
        bb = draw.textbbox((0,0), test, font=font)
        if (bb[2]-bb[0]) > max_w and cur: lines.append(cur); cur = w
        else: cur = test
    if cur: lines.append(cur)
    return lines or [text]

def put_text(img_cv, text, x1, y1, x2, y2, bg_rgb):
    if not text.strip(): return img_cv
    bw, bh = x2-x1, y2-y1; cx, cy = (x1+x2)/2, (y1+y2)/2
    lum = bg_rgb[0]*.299 + bg_rgb[1]*.587 + bg_rgb[2]*.114
    tc = (10,8,5) if lum > 100 else (240,235,215)
    pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    best_fs, best_lines, best_lh = 9, [text], 12
    for fs in range(int(bh*.52), 8, -1):
        font = get_font(fs); lines = wrap_text(draw, text, font, bw*.84); lh = fs*1.36
        if len(lines)*lh <= bh*.85: best_fs, best_lines, best_lh = fs, lines, lh; break
    font = get_font(best_fs); th = len(best_lines)*best_lh; sy = cy - th/2 + best_lh*.22
    for i, line in enumerate(best_lines):
        bb = draw.textbbox((0,0), line, font=font)
        draw.text((int(cx-(bb[2]-bb[0])/2), int(sy+i*best_lh)), line, font=font, fill=tc)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def plain_fill(img, x1, y1, x2, y2, bg):
    H,W = img.shape[:2]; x1,y1=max(0,x1),max(0,y1); x2,y2=min(W,x2),min(H,y2)
    img[y1:y2,x1:x2]=(bg[2],bg[1],bg[0]); return img

def inpaint_region(img, x1, y1, x2, y2, bg):
    H,W = img.shape[:2]; x1,y1=max(0,x1),max(0,y1); x2,y2=min(W,x2),min(H,y2)
    if x2<=x1 or y2<=y1: return img
    img[y1+1:y2-1,x1+1:x2-1]=(bg[2],bg[1],bg[0])
    mask=np.zeros((H,W),dtype=np.uint8); mask[y1+1:y2-1,x1+1:x2-1]=255
    return cv2.inpaint(img,mask,5,cv2.INPAINT_TELEA)

def contour_fill(img, x1, y1, x2, y2, bg, thresh_val=190):
    H,W=img.shape[:2]; x1,y1=max(0,x1),max(0,y1); x2,y2=min(W,x2),min(H,y2)
    bw,bh=x2-x1,y2-y1
    if bw<8 or bh<8: return img, None
    crop=img[y1:y2,x1:x2].copy()
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,thresh_val,255,cv2.THRESH_BINARY)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,k)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_DILATE,k)
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return inpaint_region(img,x1,y1,x2,y2,bg), None
    cnt=max(contours,key=cv2.contourArea)
    fill_mask=np.zeros((bh,bw),dtype=np.uint8)
    cv2.drawContours(fill_mask,[cnt],-1,255,-1)
    crop_out=crop.copy(); crop_out[fill_mask==255]=(bg[2],bg[1],bg[0])
    img[y1:y2,x1:x2]=crop_out
    full_mask=np.zeros((H,W),dtype=np.uint8)
    shifted=cnt+np.array([x1,y1])
    cv2.drawContours(full_mask,[shifted],-1,255,-1)
    eroded=cv2.erode(full_mask,np.ones((5,5),np.uint8))
    border=full_mask.copy(); border[eroded==255]=0
    img=cv2.inpaint(img,border,4,cv2.INPAINT_TELEA)
    return img, fill_mask

def place_text_in_body(img, text, x1, y1, x2, y2, fill_mask, bg):
    bh=y2-y1
    row_w=[(int(fill_mask[r].sum()//255),r) for r in range(bh)]
    row_w.sort(reverse=True)
    wide=sorted([r for _,r in row_w[:max(1,bh//2)]])
    if not wide: return put_text(img,text,x1,y1,x2,y2,bg)
    by1=y1+wide[0]; by2=y1+wide[-1]
    cols=np.where(fill_mask[wide[0]:wide[-1]+1].any(axis=0))[0]
    if not len(cols): return put_text(img,text,x1,y1,x2,y2,bg)
    bx1=x1+int(cols[0]); bx2=x1+int(cols[-1])
    pad=max(4,int((bx2-bx1)*.05))
    bx1+=pad; bx2-=pad
    by1+=max(3,int((by2-by1)*.05)); by2-=max(3,int((by2-by1)*.05))
    if bx2>bx1 and by2>by1: return put_text(img,text,bx1,by1,bx2,by2,bg)
    return put_text(img,text,x1,y1,x2,y2,bg)

def sample_bg(img, x1, y1, x2, y2, btype):
    H,W=img.shape[:2]; bw,bh=x2-x1,y2-y1
    pts=[]
    for t in [.07,.2,.35,.5,.65,.8,.93]:
        pts+=[(int(x1+bw*t),int(y1+bh*.06)),(int(x1+bw*t),int(y1+bh*.94)),
              (int(x1+bw*.05),int(y1+bh*t)),(int(x1+bw*.95),int(y1+bh*t))]
    cols=[]
    for px,py in pts:
        px=min(W-1,max(0,px)); py=min(H-1,max(0,py))
        b,g,r=[int(v) for v in img[py,px,:3]]
        cols.append((r,g,b,r+g+b))
    cols.sort(key=lambda c:c[3],reverse=True)
    top=cols[:8]
    r=int(sum(c[0] for c in top)/len(top))
    g=int(sum(c[1] for c in top)/len(top))
    b=int(sum(c[2] for c in top)/len(top))
    if btype=='dialogue' and r>180 and g>180 and b>180: return (255,255,255)
    return (r,g,b)

def apply_bubble(img, bubble, W, H):
    x1=max(0,min(W-1,int(bubble['x1']))); y1=max(0,min(H-1,int(bubble['y1'])))
    x2=max(0,min(W,int(bubble['x2']))); y2=max(0,min(H,int(bubble['y2'])))
    if x2-x1<6 or y2-y1<6: return img
    arabic=bubble.get('arabic',''); btype=bubble.get('type','dialogue')
    bg=tuple(bubble['bg_color']) if 'bg_color' in bubble else sample_bg(img,x1,y1,x2,y2,btype)
    if btype=='title':
        img=plain_fill(img,x1,y1,x2,y2,bg)
        if arabic: img=put_text(img,arabic,x1,y1,x2,y2,bg)
    elif btype=='narration':
        img=inpaint_region(img,x1,y1,x2,y2,bg)
        if arabic: img=put_text(img,arabic,x1,y1,x2,y2,bg)
    elif btype=='dialogue':
        result=contour_fill(img,x1,y1,x2,y2,bg)
        if isinstance(result,tuple):
            img,fill_mask=result
            if arabic and fill_mask is not None:
                img=place_text_in_body(img,arabic,x1,y1,x2,y2,fill_mask,bg)
            elif arabic: img=put_text(img,arabic,x1,y1,x2,y2,bg)
        else:
            img=result
            if arabic: img=put_text(img,arabic,x1,y1,x2,y2,bg)
    else:
        if arabic: img=put_text(img,arabic,x1,y1,x2,y2,bg)
    return img

CLAUDE_SYSTEM = '''أنت نظام تحليل كوميكس متخصص. استخرج كل فقاعات النص بإحداثيات بكسل دقيقة وترجمها للعربية.
أجب بـ JSON فقط:
{"bubbles":[{"id":1,"type":"dialogue","original":"...","arabic":"...","x1":120,"y1":45,"x2":380,"y2":190}]}
type: dialogue=فقاعة بيضاء | narration=صندوق بيج | title=عنوان | sfx=مؤثر صوتي
ترجمة: حوار طبيعي، سرد أدبي، مؤثرات معربة، أسماء كما تنطق'''

def call_claude(image_path, api_key, W, H, retries=3):
    import io
    pil=Image.open(image_path)
    if pil.width>1600 or pil.height>2200: pil.thumbnail((1600,2200),Image.LANCZOS)
    buf=io.BytesIO(); pil.save(buf,format='JPEG',quality=88)
    b64=base64.b64encode(buf.getvalue()).decode()
    payload=json.dumps({"model":CLAUDE_MODEL,"max_tokens":MAX_TOKENS,
        "system":CLAUDE_SYSTEM,"messages":[{"role":"user","content":[
        {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}},
        {"type":"text","text":f"الصورة {W}x{H} بكسل. استخرج كل الفقاعات بإحداثيات دقيقة وترجم. JSON فقط."}]}]}).encode()
    for attempt in range(retries):
        try:
            req=urllib.request.Request('https://api.anthropic.com/v1/messages',data=payload,
                headers={'Content-Type':'application/json','x-api-key':api_key,
                         'anthropic-version':'2023-06-01'},method='POST')
            with urllib.request.urlopen(req,timeout=90) as resp:
                data=json.loads(resp.read())
                raw=data['content'][0]['text'].strip()
                raw=re.sub(r'^```json\s*','',raw); raw=re.sub(r'^```\s*','',raw)
                raw=re.sub(r'\s*```$','',raw).strip()
                m=re.search(r'\{[\s\S]*\}',raw)
                if not m: return []
                return json.loads(m.group()).get('bubbles',[])
        except urllib.error.HTTPError as e:
            if e.code==529:
                time.sleep(20*(attempt+1))
            elif attempt>=retries-1: raise
        except Exception:
            if attempt>=retries-1: raise
            time.sleep(5)
    return []

def process_image(inp, out, api_key):
    try:
        pil=Image.open(inp).convert('RGB')
        img=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
        H,W=img.shape[:2]
        bubbles=call_claude(inp,api_key,W,H)
        if not bubbles: shutil.copy2(inp,out); return True
        for b in bubbles: img=apply_bubble(img,b,W,H)
        os.makedirs(os.path.dirname(out) or '.',exist_ok=True)
        cv2.imwrite(out,img,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
        return True
    except Exception as e:
        print(f'    Error: {e}'); return False

def extract_archive(path, dest):
    os.makedirs(dest,exist_ok=True)
    ext=Path(path).suffix.lower()
    if ext in ('.cbz','.zip'):
        with zipfile.ZipFile(path,'r') as zf: zf.extractall(dest)
    elif ext in ('.cbr','.rar'):
        try:
            import rarfile
            with rarfile.RarFile(path) as rf: rf.extractall(dest)
        except ImportError:
            if os.system(f'unrar x "{path}" "{dest}"')!=0:
                raise RuntimeError('pip install rarfile or apt install unrar')
    imgs=[]
    for root,_,files in os.walk(dest):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED_IMG:
                imgs.append(os.path.join(root,f))
    return sorted(imgs)

def repack_cbz(paths, out):
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(paths): zf.write(p,os.path.basename(p))
    print(f'  Packed -> {out}')

def process_batch(inputs, out_dir, api_key, workers=MAX_WORKERS):
    os.makedirs(out_dir,exist_ok=True)
    total=len(inputs); done=0; results={}
    print(f'\nبدء الترجمة — {total} صفحة | {workers} متوازي\n'+'-'*40)
    def job(p):
        name=Path(p).stem+'_ar.jpg'
        out=os.path.join(out_dir,name)
        ok=process_image(p,out,api_key)
        return p,out,ok
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(job,p):p for p in inputs}
        for fut in as_completed(futures):
            src,out,ok=fut.result(); done+=1
            print(f'  [{done:>3}/{total}] {"✓" if ok else "✗"} {Path(src).name}')
            results[src]=out if ok else None
    ok_n=sum(1 for v in results.values() if v)
    print(f'\nنجح: {ok_n}  |  فشل: {total-ok_n}')
    return [v for v in results.values() if v]

def main():
    ap=argparse.ArgumentParser(description='AutoScanlate AR')
    ap.add_argument('--input','-i',required=True)
    ap.add_argument('--output','-o',default='')
    ap.add_argument('--api_key','-k',default='')
    ap.add_argument('--workers','-w',type=int,default=MAX_WORKERS)
    ap.add_argument('--repack','-r',action='store_true',default=True)
    ap.add_argument('--quality','-q',type=int,default=JPEG_QUALITY)
    args=ap.parse_args()
    api_key=args.api_key or os.environ.get('ANTHROPIC_API_KEY','')
    if not api_key: print('مفتاح API مطلوب'); sys.exit(1)
    global JPEG_QUALITY; JPEG_QUALITY=args.quality
    p=Path(args.input)
    out_dir=args.output or (p.stem+'_ar')
    imgs=[]; archive=False; stem=''
    if p.is_dir():
        imgs=[str(f) for f in sorted(p.rglob('*')) if f.suffix.lower() in SUPPORTED_IMG]
    elif p.suffix.lower() in ('.cbz','.cbr','.zip','.rar'):
        archive=True; stem=p.stem
        imgs=extract_archive(str(p),os.path.join(out_dir,'_extracted'))
    elif p.suffix.lower() in SUPPORTED_IMG:
        imgs=[str(p)]
    else: print(f'نوع غير مدعوم: {p.suffix}'); sys.exit(1)
    if not imgs: print('لم يُعثر على صور'); sys.exit(1)
    if len(imgs)>100: imgs=imgs[:100]
    translated=process_batch(imgs,os.path.join(out_dir,'pages'),api_key,args.workers)
    if archive and args.repack and translated:
        repack_cbz(translated,os.path.join(out_dir,stem+'_ar.cbz'))
    print(f'الملفات في: {out_dir}/')

if __name__=='__main__':
    main()