"""Collection contract checks; no scientific model training."""
import importlib
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

C=importlib.import_module('artifacts.experiments.20261104_rare_collection.collect')


def synthetic():
    T,N=8,2
    initial=dict(timestep=np.zeros(N,np.int32),body=np.full((N,4),9,np.int32),
        ach=np.zeros((N,22),bool),inv=np.zeros((N,12),np.int32),
        light=np.ones(N,np.float32),pos=np.zeros((N,2),np.int32))
    for kind in ('zombies','cows','skeletons','arrows'):
        initial[kind+'_p']=np.full((N,1,2),99,np.int32)
        initial[kind+'_m']=np.zeros((N,1),bool)
    state={k:np.repeat(v[None],T,0) for k,v in initial.items()}
    state['timestep'][:]=np.arange(1,T+1)[:,None]
    state['ach'][1:,:,0]=True;state['inv'][1:,:,0]=1
    state['body'][2:,0,0]=0
    done=np.zeros((T,N),bool);done[2:,0]=True
    frames=np.broadcast_to(np.arange(1,T+1,dtype=np.uint8)[:,None,None,None,None],(T,N,63,63,3)).copy()
    action=np.zeros((T,N),np.int32);reward=np.zeros((T,N),np.float32);reward[1]=1
    out=(frames,action,reward,done,done.copy(),np.zeros((T,N),bool),np.full((T,N),1/17,np.float32),state)
    return np.zeros((N,63,63,3),np.uint8),initial,out


class Contracts(unittest.TestCase):
    def test_schedule_families_and_boundaries(self):
        for r,base in [(0,0),(2,.02),(4,.05),(6,.1)]:
            e,p=C.schedule(r,2500);np.testing.assert_allclose(e,base)
            self.assertEqual(p['component'],'competent')
        for r,prefix in [(1,512),(3,1024)]:
            e,p=C.schedule(r,2500)
            self.assertEqual(p['prefix'],prefix)
            np.testing.assert_allclose(e[:prefix],.02)
            np.testing.assert_equal(e[prefix:prefix+16],1)
            np.testing.assert_allclose(e[prefix+16:prefix+128],.02)
            np.testing.assert_equal(e[prefix+128:prefix+144],1)

    def test_first_death_and_survivor_both_retained(self):
        f,s,o=synthetic();eps=np.full(8,.1,np.float32)
        es,a=C.pack(f,s,o,eps,round_index=0,seed=C.SEED,
                    policy=dict(component='competent',base_epsilon=.1))
        self.assertEqual([len(e) for e in es],[3,8])
        self.assertEqual([len(e.observations) for e in es],[4,9])
        self.assertTrue(es[0].terminated[-1]);self.assertTrue(es[1].truncated[-1])
        self.assertEqual(es[0].terminal_cause,'health')
        self.assertEqual(int(es[0].observations[-1,0,0,0]),3)
        self.assertEqual(a['t_action'].shape,(11,))
        self.assertEqual(a['s_body'].shape,(13,4))
        self.assertEqual(len(set(e.episode_id for e in es)),2)

    def test_schedule_is_not_misrepresented_as_constant_epsilon(self):
        f,s,o=synthetic()
        es,a=C.pack(f,s,o,np.full(8,.02,np.float32),round_index=1,seed=C.SEED,
                   policy=dict(component='bursts',base_epsilon=.02))
        self.assertTrue(all(e.epsilon is None for e in es))
        self.assertIn('t_exogenous',a);self.assertIn('t_behavior_probability',a)

    def test_summary_and_monitor_partition(self):
        f,s,o=synthetic();es,a=C.pack(f,s,o,np.full(8,.1,np.float32),round_index=0,seed=C.SEED,
                   policy=dict(component='competent',base_epsilon=.1))
        q=C.summarize(es,a)
        self.assertEqual(q['all']['episodes'],2);self.assertEqual(q['all']['transitions'],11)
        self.assertEqual(q['all']['deaths'],1)
        self.assertEqual(q['all']['unlocks']['COLLECT_WOOD'],2)
        self.assertEqual(q['all']['inventory_increases']['0'],2)
        for key in ['episodes','transitions','deaths','damage']:
            self.assertEqual(q['all'][key],q['train'][key]+q['monitor'][key])
        self.assertEqual(len(q['all']['unlocks']),22)

    def test_saved_shard_roundtrip(self):
        f,s,o=synthetic();es,a=C.pack(f,s,o,np.full(8,.1,np.float32),round_index=0,seed=C.SEED,
                   policy=dict(component='competent',base_epsilon=.1))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'shard-000000.pt';record=C.save_episode_shard(p,es)
            self.assertEqual(record['transitions'],11)
            q=torch.load(p,weights_only=False,mmap=True)
            for e,row in zip(es,q['episodes']):
                restored=C.Episode(**row)
                self.assertTrue(torch.equal(e.observations,restored.observations))
                self.assertFalse(restored.bc_eligible)

    def test_corrupted_alignment_rejected(self):
        f,s,o=synthetic();es,a=C.pack(f,s,o,np.full(8,.1,np.float32),round_index=0,seed=C.SEED,
                   policy=dict(component='competent',base_epsilon=.1))
        a['t_reward'][0]=77
        with self.assertRaises(AssertionError):C.validate_pack(es,a)


if __name__=='__main__':unittest.main()
