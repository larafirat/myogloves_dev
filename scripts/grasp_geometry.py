"""Where does this hand actually grasp, with the wrist held still?

Drives the natural flexors from open to closed and records where the digits
sweep, so the object can be placed in the volume the hand encloses rather than
beside it.
"""
import sys, os, numpy as np, mujoco
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
hb.STABILISE_WRIST=True

ad=hb.glove_dev_adapters()['healthy_box']
# put the object far away so it cannot interfere with the free-closure measurement
ad.pos_override=(0.6,0.0,0.05)
m,d=ad.build(np.random.default_rng(1000)); mujoco.mj_forward(m,d)
hb.settle(m,d,ad.oid,ad.settle_cap)

PALM='capitate'; TIPS=['distph2','distph3','distph4','distph5','distal_thumb']
pid=m.body(PALM).id
def snap():
    return {b: d.xpos[m.body(b).id].copy() for b in TIPS}, d.xpos[pid].copy(), d.xmat[pid].reshape(3,3).copy()

open_tips, palm0, R0 = snap()
print("PALM (capitate) frame, wrist held:")
print(f"   position {np.round(palm0,3)}")
for i,ax in enumerate("xyz"):
    print(f"   local {ax} axis -> world {np.round(R0[:,i],2)}")

ramp=max(1,int(1.5/m.opt.timestep))
for k in range(int(4.0/m.opt.timestep)):
    ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
closed_tips, palm1, _ = snap()

print("\ndigit tips, world coords:")
print(f"   {'digit':14}{'OPEN':>26}{'CLOSED':>26}")
for b in TIPS:
    print(f"   {b:14}{str(np.round(open_tips[b],3)):>26}{str(np.round(closed_tips[b],3)):>26}")

cl=np.array([closed_tips[b] for b in TIPS])
op=np.array([open_tips[b] for b in TIPS])
print(f"\n   palm                     {np.round(palm1,3)}")
print(f"   closed-tip centroid      {np.round(cl.mean(axis=0),3)}")
print(f"   open-tip centroid        {np.round(op.mean(axis=0),3)}")
print(f"   palm -> closed centroid  {np.linalg.norm(cl.mean(axis=0)-palm1)*1000:.0f} mm")
print(f"   open tip spread (max pairwise) {max(np.linalg.norm(a-b) for a in op for b in op)*1000:.0f} mm")
print(f"   closed tip spread              {max(np.linalg.norm(a-b) for a in cl for b in cl)*1000:.0f} mm")
mid=(palm1+cl.mean(axis=0))/2
print(f"\n   => an object held against the palm should centre near {np.round(mid,3)}")
print(f"      (midway palm-to-closed-fingertips), radius up to "
      f"{np.linalg.norm(cl.mean(axis=0)-palm1)/2*1000:.0f} mm")
