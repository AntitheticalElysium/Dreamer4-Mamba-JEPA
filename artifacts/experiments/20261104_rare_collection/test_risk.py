import importlib
import unittest
import numpy as np

R=importlib.import_module('artifacts.experiments.20261104_rare_collection.risk_collect')
T=importlib.import_module('artifacts.experiments.20261104_rare_collection.test_collect')


class RiskContracts(unittest.TestCase):
    def test_reset_exploration_and_real_caps(self):
        for r in (0,1,6,7):
            eps,policy=R.schedule(r)
            self.assertEqual(len(eps),1024)
            self.assertEqual(policy['component'],'reset_exploration')
            np.testing.assert_equal(eps,policy['base_epsilon'])
        self.assertEqual(R.schedule(1)[1]['base_epsilon'],1)
        self.assertEqual(R.schedule(6)[1]['base_epsilon'],.25)

    def test_late_policy_switch_is_clock_based(self):
        for r,prefix in [(8,512),(17,1024)]:
            eps,policy=R.schedule(r)
            self.assertEqual(len(eps),2500)
            np.testing.assert_allclose(eps[:prefix],.02)
            np.testing.assert_equal(eps[prefix:],1)
            self.assertEqual(policy['prefix'],prefix)

    def test_uniform_flag_and_probability_contract(self):
        f,s,o=T.synthetic();o=list(o)
        o[5]=np.ones_like(o[5]);o[6]=np.full_like(o[6],1/17)
        es,a=R.C.pack(f,s,tuple(o),np.ones(8,np.float32),round_index=1,seed=R.SEED,
                       policy=dict(component='competent',base_epsilon=1.))
        self.assertTrue(a['t_exogenous'].all())
        self.assertTrue(all(e.epsilon==1. for e in es))
        self.assertEqual(int(sum(e.terminated.sum() for e in es)),1)
        self.assertTrue(es[1].truncated[-1])
        self.assertNotEqual(es[0].episode_id,f'coverage-v1:{R.C.SEED}:1:0')


if __name__=='__main__':unittest.main()
