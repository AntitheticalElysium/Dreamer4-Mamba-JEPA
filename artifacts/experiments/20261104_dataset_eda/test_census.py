import importlib
import unittest
import numpy as np

C=importlib.import_module('artifacts.experiments.20261104_dataset_eda.census')


class CensusChecks(unittest.TestCase):
    def test_terminal_suffixes_are_admitted_without_borrowing_actions(self):
        ex=np.array([1,1,0,1,1],bool)
        s,e,covered=C.exogenous_windows(ex,True,4)
        np.testing.assert_equal(s,[3,4]);np.testing.assert_equal(e,[5,5])
        np.testing.assert_equal(covered,[False,False,False,True,True])
        s,_,_=C.exogenous_windows(ex,False,4);self.assertEqual(len(s),0)

    def test_full_windows_and_last_terminal_step(self):
        ex=np.ones(5,bool)
        s,e,_=C.exogenous_windows(ex,False,4)
        np.testing.assert_equal(s,[0,1]);np.testing.assert_equal(e,[4,5])
        s,e,_=C.exogenous_windows(ex,True,4)
        np.testing.assert_equal(s,np.arange(5));np.testing.assert_equal(e,[4,5,5,5,5])

    def test_camera_scroll_does_not_count_as_terrain_edit(self):
        a={k:None for k in C.FIELDS};a['s_body']=np.full((2,4),9,np.int32)
        a['s_inv']=np.zeros((2,12),np.int32);a['s_pos']=np.array([[0,0],[0,1]])
        a['s_dir']=np.array([2,2]);a['s_light']=np.ones(2);a['s_sleep']=np.zeros(2,bool)
        a['s_ach']=np.zeros((2,22),bool);a['s_view']=np.full((2,7,9),4,np.int8)
        a['s_view'][0,3,5]=5;a['s_view'][1,:,:8]=a['s_view'][0,:,1:];a['s_view'][1,:,8]=2
        for kind in ['zombies','cows','skeletons','arrows']:
            a[f's_{kind}_p']=np.full((2,1,2),99);a[f's_{kind}_m']=np.zeros((2,1),bool)
            a[f's_{kind}_cd']=np.zeros((2,1),int)
        a['t_action']=np.array([2]);a['t_reward']=np.zeros(1);a['t_terminated']=np.zeros(1,bool)
        a['t_exogenous']=np.ones(1,bool);a['ep_T']=np.array([1])
        q,_=C.episode(a,np.array([0]),np.array([0]))
        self.assertFalse(any(k.startswith('edit/') and v['n'] for k,v in q.items()))
        a['s_view'][1,3,4]=2
        q,_=C.episode(a,np.array([0]),np.array([0]))
        self.assertEqual(q['edit/TREE_to_GRASS'],{'n':1,'episodes':1})


if __name__=='__main__':unittest.main()
