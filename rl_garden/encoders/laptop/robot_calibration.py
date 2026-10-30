"""Measured joint feedback and URDF/tool calibration; no object asset catalog."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch,yaml

def rotation(r):
    a,b,c=r;sa,sb,sc=np.sin(r);ca,cb,cc=np.cos(r)
    return np.array([[cc,-sc,0],[sc,cc,0],[0,0,1]])@np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])@np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])

def axis_rotation(axis,angle):
    axis=np.asarray(axis,dtype=float);axis=axis/np.linalg.norm(axis);x,y,z=axis
    K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)

def origin(j):
    o=j.find('origin');T=np.eye(4)
    if o is not None:
        T[:3,:3]=rotation(np.fromstring(o.get('rpy','0 0 0'),sep=' '))
        T[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
    return T

class RobotCalibration:
    def __init__(self, assets_path, gripper_calibration):
        # Same root as RoboTwin's ASSETS_PATH: it contains the assets/ directory.
        p=Path(assets_path)/'assets/embodiments/aloha-agilex';self.cfg=yaml.safe_load((p/'config.yml').read_text())
        f=p/'urdf/arx5_description_isaac.urdf';root=ET.parse(f).getroot()
        self.by_child={j.find('child').get('link'):j for j in root.findall('joint')}
        self.sha={'urdf':hashlib.sha256(f.read_bytes()).hexdigest(),'config':hashlib.sha256((p/'config.yml').read_bytes()).hexdigest()}
        self.offset=json.loads(Path(gripper_calibration).read_text())['aperture']['offset_m']
    def fk(self,names,values,side):
        q=dict(zip(names,values));link=self.cfg['move_group'][0 if side=='left' else 1];chain=[]
        while link in self.by_child:
            j=self.by_child[link];chain.append(j);link=j.find('parent').get('link')
        T=np.eye(4);axes=[]
        arm=self.cfg['arm_joints_name'][0 if side=='left' else 1]
        for j in reversed(chain):
            T=T@origin(j);kind=j.get('type');name=j.get('name')
            if kind!='fixed':
                assert name in q,'Missing measured joint '+name
                ax=np.fromstring(j.find('axis').get('xyz','1 0 0'),sep=' ')
                if name in arm:axes.append((name,T[:3,3].copy(),T[:3,:3]@ax))
                D=np.eye(4)
                if kind in ('revolute','continuous'):D[:3,:3]=axis_rotation(ax,q[name])
                elif kind=='prismatic':D[:3,3]=ax*q[name]
                else:raise ValueError(kind)
                T=T@D
        R=T[:3,:3]@np.asarray(self.cfg['global_trans_matrix'])@np.asarray(self.cfg['delta_matrix'])
        tcp=T[:3,3]+R@np.array([self.cfg['gripper_bias'],0,0])
        J=np.zeros((6,6))
        for name,p,axis in axes:
            i=arm.index(name);J[:3,i]=np.cross(axis,tcp-p);J[3:,i]=axis
        return tcp,R,J
    def feedback(self,record):
        fields={k:[] for k in ('q','dq','width','width_rate','tcp_p','tcp_R','tcp_J')}
        cmd={k:[] for k in ('reference_p','reference_R','width_target','q_target','q_target_valid')}
        for side,prefix in [('left','fl'),('right','fr')]:
            r=record[side];names=r['names'];q=np.array(r['qpos']);dq=np.array(r['qvel']);target=np.array(r['targets'])
            assert len(names)==len(q)==len(dq)==len(target) and np.isfinite([q,dq,target]).all()
            arm=[names.index(prefix+'_joint'+str(i)) for i in range(1,7)]
            fingers=[names.index(prefix+'_joint'+str(i)) for i in (7,8)]
            lo=[];hi=[]
            for i in (7,8):
                j=next(j for j in self.by_child.values() if j.get('name')==prefix+'_joint'+str(i));l=j.find('limit');lo.append(float(l.get('lower')));hi.append(float(l.get('upper')))
            # Target is the attainable joint-drive reference; not the ACT EE goal.
            p,R,J=self.fk(names,q,side);cp,cR,_=self.fk(names,target,side)
            fields['q'].extend(q[arm]);fields['dq'].extend(dq[arm])
            fields['width'].append(float(q[fingers].sum()+self.offset));fields['width_rate'].append(float(dq[fingers].sum()))
            fields['tcp_p'].append(p);fields['tcp_R'].append(R);fields['tcp_J'].append(J)
            cmd['reference_p'].append(cp);cmd['reference_R'].append(cR);cmd['width_target'].append(float(np.clip(target[fingers],lo,hi).sum()+self.offset))
            cmd['q_target'].extend(target[arm]);cmd['q_target_valid'].append(True)
        tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)[None]
        s={k:tensor(v) for k,v in fields.items()};c={k:tensor(v) for k,v in cmd.items()};c['q_target_valid']=c['q_target_valid'].bool()
        return s,c
