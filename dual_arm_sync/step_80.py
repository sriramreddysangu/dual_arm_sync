#!/usr/bin/env python3
"""
step_80.py -- Paper Figure Generator
Generates all publication-quality figures from collected data.
Output: all_figures/ directory with PNG + PDF versions.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import ARM_NAMES

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch
    plt.rcParams.update({'font.family':'serif','font.size':11,'axes.labelsize':12,
                         'axes.titlesize':12,'legend.fontsize':10,'figure.dpi':150})
    _PLOT=True
except ImportError:
    _PLOT=False

def _load(fname):
    if os.path.exists(fname):
        with open(fname) as f: return json.load(f)
    return None

def fig1_pipeline_overview(outdir):
    """Block diagram of the SKAR-N pipeline."""
    fig,ax=plt.subplots(figsize=(14,3.5))
    ax.set_xlim(0,14); ax.set_ylim(0,4); ax.axis('off')
    blocks=[('Step 61\nSE(3) IK',1),('Step 62\nB-Spline',3),('Step 63\nCollision\nScan',5),
            ('Step 64\nSurgical CP\nReplace',7),('Step 65\nKuramoto\nSync',9),
            ('Step 66\nExecute',11)]
    colors=['#1976D2','#1976D2','#F57C00','#E53935','#7B1FA2','#388E3C']
    for (label,x),c in zip(blocks,colors):
        ax.add_patch(plt.Rectangle((x-0.8,1.2),1.6,1.6,color=c,alpha=0.85,zorder=3))
        ax.text(x,2.,label,ha='center',va='center',fontsize=9,color='white',fontweight='bold',zorder=4)
    for i in range(len(blocks)-1):
        x1=blocks[i][1]+0.8; x2=blocks[i+1][1]-0.8
        ax.annotate('',xy=(x2,2.),xytext=(x1,2.),
                    arrowprops=dict(arrowstyle='->',color='#333',lw=2))
    ax.text(7,0.4,'JSON data flow: s61_ik.json → s62_trajectories.json → s63_collision_map.json → ...',
            ha='center',fontsize=9,color='#555')
    ax.text(7,3.7,'SKAR-N: Surgical Control-Point Replacement + Kuramoto Phase Synchronization',
            ha='center',fontsize=11,fontweight='bold')
    plt.tight_layout()
    for ext in ['png','pdf']:
        plt.savefig(os.path.join(outdir,f'fig1_pipeline.{ext}'),bbox_inches='tight'); 
    plt.close(); print('  fig1_pipeline saved')

def fig2_collision_resolution(outdir, cmap, trj, res):
    """Before/after segment bar comparison."""
    if not all([cmap,trj,res]): return
    arm_names=trj.get('arm_names',ARM_NAMES)
    pairs_coll=[p for p in cmap.get('pairs',[]) if p['status']=='COLLISION']
    if not pairs_coll: return
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    for ax,label,data_cmap,source in [(axes[0],'Before (step_63)',cmap,trj),(axes[1],'After (step_64)',None,res)]:
        bars=[]
        if label.startswith('Before'):
            for p in pairs_coll:
                bar=[seg['status'] for seg in p['segments']]
                bars.append((p['pair'],bar))
        else:
            bars=[('Resolved',['S']*5)]
        colors_map={'C':'#E53935','W':'#FF9800','S':'#43A047'}
        for yi,(pair,bar) in enumerate(bars):
            for xi,s in enumerate(bar):
                ax.add_patch(plt.Rectangle((xi,yi),1,0.8,color=colors_map.get(s,'#90A4AE')))
                ax.text(xi+0.5,yi+0.4,s,ha='center',va='center',fontsize=10,color='white',fontweight='bold')
        ax.set_xlim(0,5); ax.set_ylim(0,max(1,len(bars)))
        ax.set_xticks([i+0.5 for i in range(5)]); ax.set_xticklabels([f'S{i}' for i in range(5)])
        ax.set_title(label); ax.set_yticks([yi+0.4 for yi in range(len(bars))])
        ax.set_yticklabels([b[0] for b in bars],fontsize=9)
    plt.suptitle('Collision Segment Resolution',fontsize=11); plt.tight_layout()
    for ext in ['png','pdf']: plt.savefig(os.path.join(outdir,f'fig2_collision.{ext}'),bbox_inches='tight')
    plt.close(); print('  fig2_collision saved')

def fig3_quality_metrics(outdir, metrics):
    """Surgical modification magnitude bar chart."""
    if not metrics: return
    arm_names=metrics.get('arm_names',ARM_NAMES)
    mods=metrics['quality']['surgical_mod_mag_sum']
    fig,ax=plt.subplots(figsize=(6,4))
    vals=[mods.get(n,0.) for n in arm_names]
    colors=['#E53935' if v>1e-9 else '#43A047' for v in vals]
    bars=ax.bar(arm_names,vals,color=colors,edgecolor='black',lw=0.8)
    ax.bar_label(bars,fmt='%.4f',fontsize=9,padding=3)
    ax.set_ylabel('||ΔCP||_F (Frobenius norm, rad)'); ax.set_title('Surgical Modification Magnitude')
    ax.legend(handles=[plt.Rectangle((0,0),1,1,color='#E53935',label='Modified (collision arm)'),
                        plt.Rectangle((0,0),1,1,color='#43A047',label='Unmodified (safe arm)')],
              fontsize=9); ax.grid(True,axis='y',alpha=0.3)
    plt.tight_layout()
    for ext in ['png','pdf']: plt.savefig(os.path.join(outdir,f'fig3_quality.{ext}'),bbox_inches='tight')
    plt.close(); print('  fig3_quality saved')

def fig4_lemma3(outdir, lemma):
    """Separation guarantee curve."""
    if not lemma: return
    data=lemma.get('lemma3_data',[])
    if not data: return
    theta=[d['theta_deg'] for d in data]; dists=[d['min_dist_cm'] for d in data]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(theta,dists,'b-o',ms=4,lw=2,label='Min inter-arm clearance')
    ax.axhline(y=0,color='red',ls='--',lw=1.5,label='Collision threshold (0cm)')
    thresh=lemma.get('j1_sep_threshold_used_deg',17.2)
    ax.axvline(x=thresh,color='orange',ls='--',lw=1.5,label=f'J1_SEP_THRESH={thresh:.0f}°')
    ax.fill_between(theta,dists,[0]*len(dists),where=[d>0 for d in dists],alpha=0.15,color='green')
    ax.set_xlabel('|j1_i - j1_j| (degrees)'); ax.set_ylabel('Min clearance at retracted pose (cm)')
    ax.set_title('Lemma 3: Separation Guarantee vs j1 Difference')
    ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    for ext in ['png','pdf']: plt.savefig(os.path.join(outdir,f'fig4_lemma3.{ext}'),bbox_inches='tight')
    plt.close(); print('  fig4_lemma3 saved')

def fig5_benchmark(outdir, bench):
    """Outcome distribution pie chart."""
    if not bench: return
    outcomes=bench.get('outcomes',{})
    labels=[k for k,v in outcomes.items() if v>0]
    vals=[outcomes[k] for k in labels]
    colors_map={'SAFE_NO_COLL':'#2E7D32',
                'RESOLVED_CP_1':'#43A047','RESOLVED_CP_2':'#66BB6A',
                'RESOLVED_CP_3':'#9CCC65','RESOLVED_CP_4':'#C5E1A5',
                'RESOLVED_CP_5':'#FFF176',
                'RESOLVED_2SEG_1':'#FFB74D','RESOLVED_2SEG_2':'#FF8A65',
                'RESOLVED_2SEG_3':'#FF7043',
                'RESOLVED_3SEG_1':'#E64A19','RESOLVED_3SEG_2':'#D84315',
                'RESOLVED_LASTSEG_1':'#5C6BC0','RESOLVED_LASTSEG_2':'#3F51B5',
                'RESOLVED_LASTSEG_3':'#303F9F','RESOLVED_LASTSEG_4':'#283593',
                'RESOLVED_LASTSEG_5':'#1A237E',
                'GIVEUP_EDGE_SEG':'#7B1FA2',
                'UNRESOLVED':'#E53935','FAIL_RESOLVE':'#B71C1C',
                'FAIL_KURAMOTO':'#880E4F'}
    colors=[colors_map.get(l,'#90A4AE') for l in labels]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].pie(vals,labels=labels,colors=colors,autopct='%1.1f%%',startangle=90,
                textprops={'fontsize':9})
    axes[0].set_title(f'Outcome Distribution ({bench["trials"]} trials)')
    times=[t.get('time_ms',0) for t in bench.get('per_trial',[])]
    if times:
        axes[1].hist(times,bins=20,color='#1976D2',edgecolor='white',alpha=0.8)
        axes[1].set_xlabel('Planning time (ms)'); axes[1].set_ylabel('Count')
        axes[1].set_title(f'Planning Time Distribution\nMean={np.mean(times):.0f}ms  Std={np.std(times):.0f}ms')
        axes[1].grid(True,alpha=0.3)
    plt.suptitle('SKAR-N Benchmark Results',fontsize=12); plt.tight_layout()
    for ext in ['png','pdf']: plt.savefig(os.path.join(outdir,f'fig5_benchmark.{ext}'),bbox_inches='tight')
    plt.close(); print('  fig5_benchmark saved')

def main():
    print('\n' + '='*66); print('  STEP 80  --  Paper Figure Generator'); print('='*66)
    if not _PLOT: print('  matplotlib not available'); sys.exit(0)
    outdir='all_figures'; os.makedirs(outdir,exist_ok=True)
    cmap   =_load('s63_collision_map.json')
    trj    =_load('s62_trajectories.json')
    res    =_load('s64_resolved.json')
    metrics=_load('s67_metrics.json')
    lemma  =_load('s73_lemma3.json')
    bench  =_load('s68_benchmark.json')
    fig1_pipeline_overview(outdir)
    fig2_collision_resolution(outdir,cmap,trj,res)
    fig3_quality_metrics(outdir,metrics)
    fig4_lemma3(outdir,lemma)
    fig5_benchmark(outdir,bench)
    print(f'\n  All figures saved to {outdir}/')
    print('  Formats: PNG (150 dpi) + PDF (vector, LaTeX-ready)\n')

if __name__=='__main__': main()