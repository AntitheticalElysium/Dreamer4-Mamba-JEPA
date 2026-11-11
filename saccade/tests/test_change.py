"""Full-frame change grounding: region independence, actual target coverage, and historical checkpoint loading."""
import copy
import numpy as np
import pytest
import torch
from saccade import data as D,model as M,train as TR


def historical_cfg():
    """The 11-08 full-frame change recipe (P1-F5 removed the head from m0.yaml; historical checkpoints still build it)."""
    cfg=TR.load_cfg();cfg['heads']['change_map']={'target':'full_frame_a9b_v2','hidden':256,'position_dim':32,'belief_dim':128}
    cfg['heads'].pop('forecast',None);cfg['losses']['groups']=['G1','G2','G3'];return cfg


def test_change_head_covers_every_screen_cell_and_has_gradient():
    cfg=historical_cfg();head=M.ChangeMap(cfg)
    p=torch.randn(2,81,16,requires_grad=True);old=torch.randn_like(p);b=torch.randn(2,256,requires_grad=True)
    y=head(old,p,b)
    assert y.shape==(2,9,9,2)
    # Directly test the formerly omitted bottom rows and player cell, rather than a global nonzero norm.
    for r,c in [(8,8),(7,0),(3,4)]:
        g=torch.autograd.grad(y[:,r,c].sum(),p,retain_graph=True)[0]
        assert g[:,r*9+c].abs().sum()>0


def test_legacy_head_weights_load_without_modifying_existing_rows():
    head=M.ChangeMap(historical_cfg());state=copy.deepcopy(head.state_dict())
    state['cell.weight']=state['cell.weight'][:63].clone();old=state['cell.weight'].clone()
    head.load_state_dict(state)
    assert torch.equal(head.cell.weight[:63],old)
    assert torch.equal(head.cell.weight[63:],torch.zeros_like(head.cell.weight[63:]))


def test_full_frame_identity_and_translated_overlay_have_no_semantic_masks():
    rng=np.random.default_rng(42)
    cells=rng.integers(0,256,(9,9,7,7,3),dtype=np.uint8)
    frame=cells.transpose(0,2,1,3,4).reshape(63,63,3)
    c,source,valid=D.change_correspondence(np.stack([frame,frame]))
    exact=(c[1:,:,:,None]==source).all((-1,-2,-3))&valid
    assert exact.shape==(1,9,9,2) and exact.all()
    # A stationary overlay is moved among several positions, not assumed to occupy a particular screen row.
    for row,col in [(0,0),(3,4),(7,2)]:
        seq=[]
        for t in range(8):
            x=np.roll(cells,t%3,axis=1).copy();x[row:row+2,col:col+2]=cells[row:row+2,col:col+2]
            seq.append(x.transpose(0,2,1,3,4).reshape(63,63,3))
        c,source,valid=D.change_correspondence(np.stack(seq))
        exact=(c[1:,:,:,None]==source).all((-1,-2,-3))&valid
        assert exact[:,row:row+2,col:col+2,0].all()
    x=frame.copy();x[3*7,4*7]^=255
    _,(pi,ri,qi,mode)=D.pair_features(np.stack([frame,x]))
    assert len(pi)>=81 and set(zip(ri[mode==0],qi[mode==0]))=={(r,c) for r in range(9) for c in range(9)}


def test_exact_identity_never_inherits_detector_static_content_bias():
    class Biased:
        def predict(self,x):return np.full(len(x),.9)
    f=np.random.default_rng(3).integers(0,256,(63,63,3),dtype=np.uint8)
    p=D.change_probabilities(np.stack([f,f]),Biased())
    assert np.array_equal(p,np.zeros((1,9,9,2),np.float32))


def test_both_comparisons_preserve_camera_following_foreground_change():
    class Detector:
        def predict(self,x):return np.full(len(x),.9)
    background=np.random.default_rng(9).integers(0,256,(9,9,7,7,3),dtype=np.uint8)
    old=background.copy();new=np.roll(background,1,axis=1).copy()
    old[4,4]=np.array([255,0,0],np.uint8);new[4,4]=old[4,4]
    frame=lambda x:x.transpose(0,2,1,3,4).reshape(63,63,3)
    p=D.change_probabilities(np.stack([frame(old),frame(new)]),Detector())
    # Identity sees an unchanged foreground. Relative to the moving background it changed. Keep both facts;
    # selecting the best match for the center would erase the latter.
    assert p[0,4,4,0]==0 and p[0,4,4,1]>.8


def test_target_store_covers_player_and_bottom_rows(tmp_path,monkeypatch):
    class Detector:
        def predict(self,x):return np.full(len(x),.7)
    f=np.zeros((3,63,63,3),np.uint8);f[1,21,28]=255;f[2]=f[1];f[2,56,56]=255
    monkeypatch.setattr(D.RP,'episode_fields',lambda shard,slot:{'observations':torch.from_numpy(f)})
    out=D.change_targets([D.Episode(0,0,2,'test')],tmp_path,Detector())
    assert out.shape==(3,9,9,2)
    assert np.isnan(out[0]).all() and np.isfinite(out[1:,:,:,0]).all()
    assert out[1,3,4,0]>0 and out[2,8,8,0]>0


def test_target_directory_separates_historical_store():
    cfg=TR.load_cfg()
    assert D.targets_dir(cfg)!=D.ROOT/'artifacts/saccade_m0/targets'


def test_legacy_target_arrays_are_rejected(tmp_path):
    with pytest.raises(ValueError,match='9x9'):
        D.validate_change_targets(np.zeros((3,7,9)),[D.Episode(0,0,2,'test')],tmp_path)


def test_refit_cannot_overwrite_a_completed_store_teacher(tmp_path):
    (tmp_path/'change_teacher.txt').write_text('historical teacher')
    np.save(tmp_path/'change_targets.npy',np.zeros((3,9,9,2)))
    with pytest.raises(FileExistsError,match='existing target store'):
        D.fit_teacher(tmp_path)
    assert (tmp_path/'change_teacher.txt').read_text()=='historical teacher'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='needs CUDA Mamba')
def test_hud_only_change_loss_trains_perception_and_memory():
    cfg=historical_cfg();torch.manual_seed(17);model=M.M0(cfg).cuda()
    eps=D.load_split(cfg)['train'][:4];offsets=D.frame_offsets(eps)
    stream=D.Streams(eps,2,8,17,targets_offsets=offsets);stream.next(8);chunk=stream.next(12)
    targets=np.full((int(offsets[-1]),9,9,2),np.nan,np.float16)
    # Only a formerly excluded location contributes: map cells cannot make this check pass.
    targets[:,7,1,0]=.7
    terms,_,_=TR.p1_losses(model,chunk,model.belief.initial(2,'cuda'),torch.zeros(2,9,256,device='cuda'),targets,cfg,np.zeros(2,bool),0)
    terms['change'].backward()
    norm=lambda module:sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None)
    assert norm(model.perception)>0 and norm(model.belief)>0 and norm(model.belief.percept_in)>0
    assert norm(model.denoiser)==0
