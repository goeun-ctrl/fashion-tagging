"""CPU-only comparison against the unmodified EVA package reader and replay code."""
from pathlib import Path
import argparse,csv,gzip,hashlib,importlib.util,json,sys
from unittest.mock import patch
import numpy as np
from PIL import Image

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(root,eva_repo,audit):
    sys.path.insert(0,str(eva_repo.resolve()))
    from experiments.role3_eva02.retrain_common_data import package as pkg,batches
    lock=json.loads((root/'shared-lock.json').read_text(encoding='utf-8'))
    pkg.check_lock(lock,'partial');rows,y=pkg.load_training(root,lock)
    probabilities,counts,hist=pkg.check_policy(root,lock,rows,y)
    plans=[pkg.load_plan(root,lock,e,probabilities) for e in range(1,31)]
    files=json.loads((root/'package-files.sha256.json').read_text(encoding='utf-8'))
    for n,h in files.items():assert sha(root/n)==h,n
    ext=[i for i,r in enumerate(rows) if r['source']=='external']
    assert len(rows)==43703 and len(ext)==506
    paths=[root/'selected_external/images'/r['relative_path'] for r in rows]
    original=json.loads(gzip.decompress(audit.read_bytes()))
    original_hashes={r['file_name']:r['sha256'] for r in original['hashes']}
    all_hashes=set(original_hashes.values())
    assert len(all_hashes)==46780,'Duplicate original bytes; must not silently change fixed splits'
    for i,r in enumerate(rows):
        if i<43197:assert r['image_sha256']==original_hashes[r['image_id']]
        else:
            h=sha(paths[i]);assert h==r['image_sha256'] and h not in all_hashes
            all_hashes.add(h)
    assert [r['image_id'] for r in rows[:43197]]==original['train_inner']
    spec=importlib.util.spec_from_file_location('source_aug',root/'original_augmentation.py')
    aug=importlib.util.module_from_spec(spec);spec.loader.exec_module(aug)
    pix=[];idx,flip,factors,order=plans[0]
    # Exercise every external image, using coefficients from actual published draws.
    ep=(np.array(ext),flip[:506],factors[:506],order[:506])
    ds=batches.PlannedDataset(paths,y,ep)
    for k,i in enumerate(ext):
        with Image.open(paths[i]) as im:
            before=np.asarray(aug.transform_rgb(im,bool(ep[1][k]),ep[2][k],ep[3][k]))
        captured=[]
        def capture(image,*args,**kw):
            arr=np.asarray(image).copy();captured.append(arr);return arr
        with patch.object(batches,'preprocess',capture):ds[k]
        assert np.array_equal(before,captured[0]),rows[i]['image_id']
        # Full normalized EVA tensor uses the same pipeline at 378 pixels.
        actual,target=ds[k]
        with Image.open(paths[i]) as im:
            expected=aug.tensor_array(im,378,batches.MEAN,batches.STD,bool(ep[1][k]),ep[2][k],ep[3][k])
        assert np.array_equal(actual.numpy(),expected) and np.isfinite(expected).all()
        assert np.array_equal(target.numpy(),y[i])
    # Full 30-epoch legacy/revised semantic identity, not merely compatible keys.
    legacy_root=root.parents[1]/'external_augmentation_20261104/prepared_v1'
    equivalent=True
    for e,plan in enumerate(plans,1):
        with np.load(legacy_root/f'epoch_plans/epoch_{e:02d}.npz',allow_pickle=False) as old:
            equivalent &= all(np.array_equal(a,old[k]) for a,k in zip(plan,['image_index','flip','factors','order']))
    assert equivalent
    complete_rejected=False
    try:pkg.check_labels(y,'complete')
    except ValueError:complete_rejected=True
    assert complete_rejected
    result=dict(package_contract='PASS',eva_reader='PASS',label_mode='partial',rows=len(rows),external=506,epochs=30,draws_per_epoch=43197,categories=27,all_file_hashes_checked=len(files),external_image_hashes_checked=506,original_cached_image_hashes_checked=46780,external_images_pre_resize_pixel_equal=506,external_images_eva_378_tensor_equal=506,legacy_epoch_selections_and_transforms_equal=True,complete_mode_correctly_rejected=True,weight_histogram=hist,known_positive_counts=counts,server_full_preflight='NOT_RUN',server_reason='Original train/val image paths and official EVA weights are not present in this local package. Actual server must run EVA preflight; no GPU training or inference was performed.',original_evidence='Cached server read-only image audit; no fresh server scan in this validation.',actual_model_comparison='NOT_RUN; no new EVA or ConvNeXt training results yet')
    (root/'compatibility_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['known_positive_counts','weight_histogram']},ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True,type=Path);p.add_argument('--eva-repo',required=True,type=Path);p.add_argument('--image-audit',required=True,type=Path);a=p.parse_args();run(a.package,a.eva_repo,a.image_audit)
