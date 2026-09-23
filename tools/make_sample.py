import subprocess, soundfile as sf, numpy as np, json
turns = [
 ("Daniel","Good morning everyone, thanks for joining the call. Today we want to go over the budget for the next quarter and figure out where the money is going."),
 ("Samantha","Thanks Daniel. I looked at the numbers last night, and honestly the travel line is the one that worries me the most. It went up nearly forty percent."),
 ("Fred","I can explain part of that. We sent three people to the conference in Denver, and the hotel prices were much higher than we expected."),
 ("Daniel","Okay, that makes sense. Do we expect the same thing next quarter, or was that a one time event?"),
 ("Samantha","Mostly one time. But we do have the regional meetings coming up in the spring, so I would not cut the line too much."),
 ("Fred","Agreed. What if we cap it at the current level and ask people to book earlier?"),
 ("Karen","Sorry I am late. I was stuck in another meeting. Did I miss the discussion about software licenses?"),
 ("Daniel","Not yet Karen, we were just finishing travel. Go ahead and tell us what you found."),
 ("Karen","We are paying for about twenty seats we never use. If we trim those, we save a few thousand dollars a year without anyone noticing."),
 ("Samantha","That is great. Can you send me the list so I can double check with the team leads before we cancel anything?"),
 ("Karen","Sure, I will send it this afternoon."),
 ("Fred","One more thing on hardware. Several laptops are more than five years old and they are getting very slow."),
 ("Daniel","Let us put that on the agenda for next week. I think we are out of time. Thanks everyone."),
]
sr=16000; out=[]; truth=[]; t=0.0
voice_ids={}
for i,(v,text) in enumerate(turns):
    fn=f"t{i}.wav"
    subprocess.run(["say","-v",v,"-o",fn,"--data-format=LEI16@16000",text],check=True)
    a,_=sf.read(fn,dtype="float32")
    # trim silence
    nz=np.nonzero(np.abs(a)>0.01)[0]; a=a[nz[0]:nz[-1]+1]
    gap=np.zeros(int(sr*0.4),dtype="float32")
    out+= [a,gap]; truth.append({"speaker":v,"start":round(t,2),"end":round(t+len(a)/sr,2)}); t+= (len(a)+len(gap))/sr
x=np.concatenate(out); sf.write("meeting.wav",x,sr)
json.dump(truth,open("truth.json","w"),indent=1)
print(len(x)/sr,"seconds"); [print(r) for r in truth]
