# -*- coding: utf-8 -*-
"""Generate one city layout (semantic map seg + height field hf) with the Unbounded Layout Generator of CityDreamer (Xie et al., CVPR 2024; github.com/hzxie/CityDreamer)
and save it as npy + preview; gen_import.py citydreamer then vectorises it into building/roads.shp.
Uses only the VQVAE + Sampler networks (pure PyTorch, no CUDA extensions of the repository); the height field is the raw generator output, without the
rendering smoothing of inference.py that resets buildings lower than 50 m to a random 50-150 m.
Environment: a CUDA build of torch; the LayoutGen.pth weights are downloaded manually from gateway.infinitescript.com (captcha).
Usage: python gen_citydreamer.py --repo dataset/generators/CityDreamer --ckpt <LayoutGen.pth> --out dataset/generated/citydreamer_s0 [--size 1536] [--seed 0] [--steps 32]
Classes: 0 NULL, 1 ROAD, 2 BLD_FACADE, 3 GREEN_LANDS, 4 CONSTRUCTION, 5 COAST_ZONES, 6 OTHERS, 7 BLD_ROOF (training data are OSM zoom-15 rasters, ~3.7 m/px at 40N; heights from the OSM height tag, m)."""
import os, sys, math, json, argparse, numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--ckpt',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--size',type=int,default=1536,help='layout side length (px)'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--temperature',type=float,default=1.0)
    a=ap.parse_args(); repo=os.path.abspath(a.repo); sys.path.insert(0,repo)
    import torch; from tqdm import tqdm
    import models.vqgan, models.sampler, utils.helpers
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev='cuda' if torch.cuda.is_available() else 'cpu'; print('device',dev, torch.cuda.get_device_name(0) if dev=='cuda' else '')
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    vqae=models.vqgan.VQAutoEncoder(ck['cfg']); sampler=models.sampler.AbsorbingDiffusionSampler(ck['cfg'])
    strip=lambda sd:{(k[7:] if k.startswith('module.') else k):v for k,v in sd.items()}   # weights saved from DataParallel carry a module. prefix
    for m,sd,nm in ((vqae,strip(ck['vqae']),'vqae'),(sampler,strip(ck['sampler']),'sampler')):
        r=m.load_state_dict(sd,strict=False); print(f'{nm}: missing {len(r.missing_keys)} unexpected {len(r.unexpected_keys)}')
        assert len(r.missing_keys)==0, r.missing_keys[:5]
    vqae=vqae.to(dev).eval(); sampler=sampler.to(dev).eval()
    cfg=ck['cfg']; OUT=cfg.NETWORK.VQGAN.RESOLUTION; FEAT=cfg.NETWORK.VQGAN.ATTN_RESOLUTION; SCALE=OUT//FEAT; STRIDE=int(FEAT*0.75)
    HALF=(FEAT-STRIDE)//2*SCALE; NEMB=cfg.NETWORK.VQGAN.N_EMBED; NCH=7
    win=int(math.ceil((a.size/SCALE-FEAT)/STRIDE)+1); csz=(win-1)*STRIDE+FEAT; osz=csz*SCALE
    print(f'patch {OUT}px, {FEAT}x{FEAT} tokens, window {win}x{win} -> {osz}px')
    code=torch.ones((1,csz,csz),device=dev).long()*NEMB
    layout=torch.zeros(1,NCH,osz,osz)
    codebook=vqae.quantize.get_codebook()
    with torch.no_grad():
        for p in tqdm(range(win*win)):
            i,j=p//win,p%win; crs,ccs=i*STRIDE,j*STRIDE
            c=code[:,crs:crs+FEAT,ccs:ccs+FEAT].reshape(1,FEAT**2)
            if torch.count_nonzero(c==NEMB): c=sampler.sample(1,a.steps,x_t=c,temperature=a.temperature,device=dev)
            one=torch.zeros((1,cfg.NETWORK.SAMPLER.BLOCK_SIZE,NEMB),device=dev); one.scatter_(2,c.unsqueeze(2),1)
            q=torch.matmul(one.view(-1,NEMB),codebook).float().view(1,FEAT,FEAT,cfg.NETWORK.VQGAN.EMBED_DIM).permute(0,3,1,2).contiguous()
            lay=vqae.decode(q).cpu()
            code[:,crs:crs+FEAT,ccs:ccs+FEAT]=c.reshape(1,FEAT,FEAT)
            lrs,lcs=crs*SCALE,ccs*SCALE
            layout[...,lrs:lrs+OUT-HALF,lcs:lcs+OUT-HALF]=lay[...,HALF:,HALF:]
    valid=layout.size(2)-HALF; crop=min(valid,a.size); layout=layout[...,:crop,:crop]
    hf=(layout[0,0]*640).numpy().astype(np.float32)
    seg=utils.helpers.onehot_to_mask(layout[[0],1:],cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES).squeeze().numpy().astype(np.int32)
    os.makedirs(a.out,exist_ok=True); np.save(f'{a.out}/hf.npy',hf); np.save(f'{a.out}/seg.npy',seg)
    cnt={int(k):int(v) for k,v in zip(*np.unique(seg,return_counts=True))}
    json.dump({'source':'CityDreamer Unbounded Layout Generator (LayoutGen.pth)','size_px':int(crop),'seed':a.seed,'steps':a.steps,'temperature':a.temperature,
               'class_pixels':cnt,'hf_bld_p50':float(np.median(hf[seg==2])) if (seg==2).any() else None},open(f'{a.out}/layout_meta.json','w'),indent=1)
    try:
        import matplotlib; matplotlib.use('agg'); import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cm=ListedColormap(['#ffffff','#9e9e9e','#3b3b7a','#8bc34a','#e0a458','#4fc3f7','#eeeeee','#1f1f4d'])
        fig,ax=plt.subplots(1,2,figsize=(14,7)); ax[0].imshow(seg,cmap=cm,vmin=0,vmax=7,interpolation='nearest'); ax[0].set_title('semantic map'); ax[0].set_axis_off()
        h=np.where(seg==2,hf,np.nan); im=ax[1].imshow(h,cmap='viridis'); ax[1].set_title('building height field (m)'); ax[1].set_axis_off(); plt.colorbar(im,ax=ax[1],shrink=.7)
        plt.savefig(f'{a.out}/layout_preview.png',dpi=100,bbox_inches='tight')
    except Exception as e: print('preview failed',e)
    print(f'[citydreamer] {crop}px layout, classes {cnt} -> {a.out}')
if __name__=='__main__': main()
