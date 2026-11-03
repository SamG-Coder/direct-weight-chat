"""Render a paced chat-UI film from unmodified, captured model responses.

Pillow draws the presentation; FFmpeg encodes it. No answers are generated,
selected, corrected or rewritten here. This is explicitly a replay, not footage
claiming live inference speed. The interactive companion uses the same transcript.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
W,H,FPS=1920,1080,30
BG='#090f16';PANEL='#111c26';LINE='#293a47';WHITE='#eaf3f5';MUTED='#9aafbd';MINT='#91ebc5';RED='#ffb3a6'
FONTS={}


def font(size,bold=False,mono=False):
    key=(size,bold,mono)
    if key not in FONTS:
        name='consola.ttf' if mono else 'seguisb.ttf' if bold else 'segoeui.ttf'
        FONTS[key]=ImageFont.truetype('C:/Windows/Fonts/'+name,size)
    return FONTS[key]


def txt(d,xy,s,size=28,color=WHITE,bold=False,mono=False):
    d.text(xy,s,font=font(size,bold,mono),fill=color,anchor='lt')


def wrap(d,s,size,width,bold=False):
    lines=[]
    for paragraph in s.split('\n'):
        line=''
        for word in paragraph.split():
            attempt=(line+' '+word).strip()
            if line and d.textlength(attempt,font=font(size,bold))>width:
                lines.append(line);line=word
            else:line=attempt
        lines.append(line)
    return lines


def paragraph(d,xy,s,size,width,color=WHITE,bold=False,gap=9):
    x,y=xy
    for line in wrap(d,s,size,width,bold):
        txt(d,(x,y),line,size,color,bold);y+=size+gap
    return y


def pill(d,xy,label,color=MINT,size=18):
    x,y=xy;tw=d.textlength(label,font=font(size,True));width=tw+28
    d.rounded_rectangle((x,y,x+width,y+size+20),radius=8,fill='#172b28',outline='#315747')
    txt(d,(x+14,y+9),label,size,color,True)
    return width


NOTES=[
 ['Uses an unfamiliar entity from the prompt.','No supporting fact: the answer stays unknown.','A new fact becomes available in the same conversation.','The later correction replaces the earlier swimming fact.','The separate flying fact is preserved.'],
 ['Without the premise, the conclusion is unknown.','The supplied premise now supports the first conclusion.','The second rule carries a negative conclusion.','The question asks about that negative conclusion.'],
 ['Subtracting a negative reaches the upper output boundary.','Adding two negative operands reaches the lower boundary.'],
 ['Failure: the older valid rule should make this answer Yes.','Failure: no story is generated. Useful chat was not achieved.']
]
DURATIONS=[[6]*5,[10,6,6,6],[5,5],[11,8]]


class Film:
    def __init__(self,data):
        self.data=data;self.events=[];t=7.0
        for ci,c in enumerate(data['conversations']):
            for ti,turn in enumerate(c['turns']):
                seconds=DURATIONS[ci][ti]
                self.events.append(dict(start=t,end=t+seconds,chapter=ci,turn=ti));t+=seconds
        self.outro=t;self.duration=t+11

    def shell(self,t):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        d.rounded_rectangle((44,36,1876,1044),radius=30,fill=PANEL,outline=LINE,width=2)
        for row in range(3):
            for col in range(3):d.rounded_rectangle((80+col*13,76+row*13,87+col*13,83+row*13),radius=2,fill=MINT)
        txt(d,(142,72),'direct-weight-chat',35,WHITE,True)
        pill(d,(492,73),'v17a / COMPACT',size=16)
        d.ellipse((1492,86,1500,94),fill=RED)
        txt(d,(1516,78),'EXPERIMENT FINISHED',18,RED,True)
        d.line((44,140,1876,140),fill=LINE,width=2)
        d.rounded_rectangle((80,1023,1840,1027),radius=2,fill='#263943')
        d.rounded_rectangle((80,1023,80+1760*min(1,t/self.duration),1027),radius=2,fill=MINT)
        return im,d

    def intro(self,t):
        im,d=self.shell(t)
        pill(d,(104,203),'ANALYTICALLY CONSTRUCTED WEIGHTS',size=18)
        paragraph(d,(104,287),'Can a chat model work\nwith no training?',79,1600,WHITE,True,gap=16)
        txt(d,(108,517),'Directly written weights. Mostly zeros. Now stored compactly.',32,MUTED)
        for x,big,small in [(108,'1,528,592','packed weight bytes'),(686,'995,905','nonzero coefficients'),(1260,'0','training steps')]:
            txt(d,(x,654),big,65,MINT,True);txt(d,(x,741),small,24,MUTED)
        d.line((104,835,1816,835),fill=LINE,width=2)
        txt(d,(108,875),'Actual model responses. Continued conversations. Failures included.',28,WHITE)
        txt(d,(108,925),'Paced replay for readability — not a live inference-speed demonstration.',23,MUTED)
        return im

    def sidebar(self,d,ci):
        d.line((432,140,432,1004),fill=LINE,width=2)
        txt(d,(80,183),'DIRECTLY WRITTEN WEIGHTS',17,MUTED,True)
        txt(d,(77,225),'1,528,592',52,WHITE,True)
        txt(d,(82,312),'bytes of losslessly packed weights',21,MUTED)
        txt(d,(82,345),'995,905 nonzero coefficients',21,MUTED)
        txt(d,(82,397),'NO TRAINING / NO PRETRAINED WEIGHTS',13,MINT,True)
        labels=['Facts, memory\n& corrections','Rules & short\nreasoning chains','Bounded signed\narithmetic','Limits &\nfailed requests']
        for i,label in enumerate(labels):
            y=459+i*99;active=i==ci
            d.rounded_rectangle((70,y,402,y+84),radius=12,fill='#1b3731' if active else PANEL,outline='#365c4d' if active else PANEL,width=1)
            txt(d,(86,y+19),f'{i+1:02}',17,MINT if active else MUTED,mono=True)
            paragraph(d,(125,y+14),label,23,260,MINT if active else MUTED,gap=6)
        paragraph(d,(82,892),'Useful general chat\nwas not achieved.',23,315,RED,gap=7)
        txt(d,(82,966),'v17a COMPACT / SAME WEIGHTS',15,MUTED)

    def conversation(self,t,event):
        ci,ti=event['chapter'],event['turn'];c=self.data['conversations'][ci];current=c['turns'][ti];age=t-event['start']
        im,d=self.shell(t);self.sidebar(d,ci)
        txt(d,(486,181),c['title'],36,WHITE,True)
        txt(d,(488,234),c['subtitle'],23,MUTED)
        pill(d,(1674,179),'REPLAY',size=16)
        items=[]
        for i in range(max(0,ti-1),ti+1):
            r=c['turns'][i];items.append(dict(role='user',text=r['prompt'],current=i==ti))
            delay=4.5 if len(r['prompt'])>100 else 1.7
            full=i<ti or age>delay+1.1
            visible=r['text'] if full else '' if age<delay else r['text'][:max(0,int((age-delay)*33))]
            items.append(dict(role='assistant',text=visible,full_text=r['text'],current=i==ti,thinking=i==ti and age<delay))
        area=Image.new('RGB',(1328,564),PANEL);ad=ImageDraw.Draw(area);layouts=[]
        for item in items:
            target=item.get('full_text',item['text']);lines=wrap(ad,target,31,1116)
            h=max(80,len(lines)*41+36);layouts.append((item,lines,h+42))
        total=sum(row[2] for row in layouts)+18*(len(layouts)-1);y=max(0,564-total)
        if total>564:y=564-total
        for item,lines,height in layouts:
            user=item['role']=='user';x=110 if user else 0;boxw=1218 if user else 1180
            txt(ad,(x+boxw-42-ad.textlength('YOU',font=font(15,True)) if user else x+3,y+2),'YOU' if user else 'V17A · FROZEN WEIGHTS',15,MUTED,True)
            ad.rounded_rectangle((x,y+29,x+boxw,y+height),radius=17,fill='#243642' if user else '#162e27',outline='#38505f' if user else '#34594a',width=1)
            if item.get('thinking',False):
                for j in range(3):
                    brightness=int(110+60*(.5+.5*math.sin(age*5-j)))
                    ad.ellipse((x+27+j*18,y+58,x+34+j*18,y+65),fill=(brightness,brightness+25,min(255,brightness+32)))
            else:paragraph(ad,(x+25,y+48),item['text'],31,1116,WHITE if user else '#c9f5df',gap=10)
            y+=height+18
        im.paste(area,(486,282))
        d.rounded_rectangle((486,875,1818,952),radius=12,fill='#101b25',outline=LINE,width=1)
        delay=4.5 if len(current['prompt'])>100 else 1.7
        note=NOTES[ci][ti] if age>delay else 'The same conversation context is retained between these turns.'
        txt(d,(508,896),note,24,RED if ci==3 and age>delay else MUTED)
        txt(d,(488,977),'ACTUAL MODEL OUTPUT · PACED REPLAY · OFFLINE INFERENCE',16,MUTED,True)
        txt(d,(1610,975),f'Turn {ti+1} of {len(c["turns"])}',19,MINT)
        return im

    def ending(self,t):
        im,d=self.shell(t)
        pill(d,(104,197),'FINAL RESULT',size=18)
        txt(d,(100,280),'Useful general chat?',76,WHITE,True)
        txt(d,(108,389),'Not achieved.',90,RED,True)
        txt(d,(108,521),'Narrow neural circuits worked. The original objective was not met.',31,MUTED)
        d.rounded_rectangle((106,609,942,799),radius=18,fill='#173028',outline='#345848')
        txt(d,(135,638),'MEASURED CAPABILITIES',18,MINT,True)
        paragraph(d,(135,683),'Context facts & corrections\nBounded arithmetic · Short rule chains',28,770,WHITE,gap=14)
        d.rounded_rectangle((971,609,1816,799),radius=18,fill='#302329',outline='#62434a')
        txt(d,(1002,638),'LIMITATIONS RETAINED',18,RED,True)
        paragraph(d,(1002,683),'Alternative proof search: 29 / 104\nOpen-ended writing & code: not achieved',28,780,WHITE,gap=14)
        txt(d,(108,852),'SOURCE + EVERY VERSION’S RESULTS',17,MUTED,True)
        txt(d,(108,887),'github.com/SamG-Coder/direct-weight-chat',28,MINT)
        txt(d,(108,940),'Model + runtime: huggingface.co/SamGCoder/direct-weight-chat-v17a',25,WHITE)
        return im

    def frame(self,t):
        if t<7:return self.intro(t)
        if t>=self.outro:return self.ending(t)
        e=next(e for e in self.events if e['start']<=t<e['end'])
        return self.conversation(t,e)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'build/showcase')
    p.add_argument('--preview-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'publication/demo-transcript-compact.json';data=json.loads(source.read_text(encoding='utf-8'));film=Film(data)
    times=[3,29,60,72,86,98];previews=[]
    for i,t in enumerate(times):
        path=a.output/f'preview-{i:02}.png';film.frame(t).save(path);previews.append(path)
    contact=Image.new('RGB',(1440,1215),BG)
    for i,path in enumerate(previews):contact.paste(Image.open(path).resize((720,405),Image.Resampling.LANCZOS),((i%2)*720,(i//2)*405))
    contact.save(a.output/'contact-sheet.jpg',quality=94)
    film.ending(film.duration).save(a.output/'poster.png')
    if a.preview_only:
        print(f'Prepared {len(previews)} preview frames; film duration {film.duration:.1f}s');return
    output=a.output/'direct-weight-chat-v17a-compact.mp4'
    if output.exists():raise SystemExit('Refusing to overwrite video')
    command=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','h264_nvenc','-preset','p6','-tune','hq','-cq','18','-b:v','0','-pix_fmt','yuv420p','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i in range(round(film.duration*FPS)):
            process.stdin.write(film.frame(i/FPS).tobytes())
            if i%(FPS*10)==0:print(f'Rendered {i/FPS:.0f}/{film.duration:.0f}s',flush=True)
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg encode failed')
    except Exception:
        process.kill();raise
    record=dict(video=output.name,width=W,height=H,fps=FPS,duration_seconds=film.duration,
                transcript_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model_sha256=data['model_sha256'],
                turns=sum(len(c['turns']) for c in data['conversations']),
                presentation='Programmatically rendered chat UI replay of the exact captured responses; pacing edited for readability. Not screen-recorded live inference. No response text changed.',
                video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),video_bytes=output.stat().st_size)
    (ROOT/'publication/video-render-compact.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
