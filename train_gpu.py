import argparse,json,random,csv
from pathlib import Path
import numpy as np, torch, yaml
from torch import nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    # modify:training-portability - TensorBoard is optional for training/metrics export.
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from tqdm import tqdm
from data import load_splits,WindowDataset
from models import MODEL_CLASSES
from metrics import regression_metrics
from visualization import save_loss_curve,save_forecast

ROOT=Path(__file__).resolve().parent

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def inv_target(z,scaler,target_idx):
    return z*scaler.scale_[target_idx]+scaler.mean_[target_idx]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default=None); ap.add_argument("--models",nargs="*",default=None)
    ap.add_argument("--epochs",type=int,default=None); ap.add_argument("--batch-size",type=int,default=None)
    ap.add_argument("--stride",type=int,default=None)
    args=ap.parse_args()
    cfg=yaml.safe_load(open(ROOT/"configs/experiment.yaml",encoding="utf-8"))
    gpu=yaml.safe_load(open(ROOT/"configs/gpu_config.yaml",encoding="utf-8"))
    seed_all(cfg["seed"])
    device=torch.device(args.device or (gpu["device"] if torch.cuda.is_available() else "cpu"))
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=bool(gpu.get("tf32",True)); torch.backends.cudnn.allow_tf32=True
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory/1024**3:.2f} GB")
    else: print("WARNING: CUDA unavailable; using CPU")
    df,raw,splits,scaler=load_splits(ROOT/"data/ETTh1.csv",cfg["features"],cfg["target"],cfg["split"],
                                     cfg["input_len"],cfg["pred_len"],args.stride or cfg["stride"])
    print(f"Rows: {len(df)} | split rows: {[len(x) for x in raw]} | windows: {[len(x[0]) for x in splits]}")
    out=ROOT/"outputs"; (out/"checkpoints").mkdir(parents=True,exist_ok=True); (out/"plots").mkdir(parents=True,exist_ok=True)
    (out/"tensorboard").mkdir(parents=True,exist_ok=True)
    # persistence baseline in original scale
    test_raw=raw[2]; ot_idx=cfg["features"].index(cfg["target"])
    Xraw=test_raw[cfg["features"]].values
    ys=[]; ps=[]; last=len(Xraw)-cfg["input_len"]-cfg["pred_len"]+1
    for s in range(0,max(0,last),args.stride or cfg["stride"]):
        ys.append(Xraw[s+cfg["input_len"]:s+cfg["input_len"]+cfg["pred_len"],ot_idx])
        ps.append(np.repeat(Xraw[s+cfg["input_len"]-1,ot_idx],cfg["pred_len"]))
    baseline=regression_metrics(np.array(ys),np.array(ps))
    baseline["Model"]="Persistence"; baseline["BestEpoch"]=0
    rows=[baseline]; json_runs={"Persistence":baseline}
    names=args.models or list(cfg["models"].keys())
    amp=bool(gpu.get("amp",True)) and device.type=="cuda"
    for name in names:
        if name not in MODEL_CLASSES: raise ValueError(f"Unknown model {name}")
        mc=cfg["models"][name].copy()
        if args.epochs: mc["epochs"]=args.epochs
        if args.batch_size: mc["batch_size"]=args.batch_size
        train_ds=WindowDataset(*splits[0]); val_ds=WindowDataset(*splits[1]); test_ds=WindowDataset(*splits[2])
        loader_kw={"num_workers":gpu.get("num_workers",0),"pin_memory":bool(gpu.get("pin_memory",True)) and device.type=="cuda"}
        tr=DataLoader(train_ds,batch_size=mc["batch_size"],shuffle=True,**loader_kw)
        va=DataLoader(val_ds,batch_size=mc["batch_size"],shuffle=False,**loader_kw)
        te=DataLoader(test_ds,batch_size=mc["batch_size"],shuffle=False,**loader_kw)
        # modify:four-model-training - architecture hyperparameters come from config;
        # training compares four candidates while Agent deployment remains single-model.
        common_keys = {"lr", "batch_size", "epochs", "patience"}
        model_kwargs = {k: v for k, v in mc.items() if k not in common_keys}
        model=MODEL_CLASSES[name](cfg["input_len"],len(cfg["features"]),cfg["pred_len"],**model_kwargs).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=mc["lr"],weight_decay=cfg["optimizer"]["weight_decay"])
        loss_fn=nn.MSELoss(); scaler_amp=torch.amp.GradScaler("cuda",enabled=amp)
        writer=SummaryWriter(str(out/"tensorboard"/name))
        best=float("inf"); best_epoch=0; wait=0; history={"train":[],"val":[]}
        ckpt=out/"checkpoints"/f"{name}_best.pt"
        for epoch in range(1,mc["epochs"]+1):
            model.train(); total=0; count=0
            for xb,yb in tr:
                xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                    pred=model(xb); loss=loss_fn(pred,yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["optimizer"]["grad_clip"])
                scaler_amp.step(opt); scaler_amp.update()
                total+=loss.item()*len(xb); count+=len(xb)
            train_loss=total/count
            model.eval(); total=0; count=0
            with torch.no_grad():
                for xb,yb in va:
                    xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
                    with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp): loss=loss_fn(model(xb),yb)
                    total+=loss.item()*len(xb); count+=len(xb)
            val_loss=total/count; history["train"].append(train_loss); history["val"].append(val_loss)
            writer.add_scalar("loss/train",train_loss,epoch); writer.add_scalar("loss/val",val_loss,epoch)
            print(f"{name:7s} epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")
            if val_loss<best-1e-7:
                best=val_loss; best_epoch=epoch; wait=0
                torch.save({"model_name":name,"model_state":model.state_dict(),"config":cfg,
                            "model_config":mc,"scaler_mean":scaler.mean_.tolist(),
                            "scaler_scale":scaler.scale_.tolist(),"target_idx":ot_idx,
                            "best_val_loss":best,"best_epoch":best_epoch},ckpt)
            else:
                wait+=1
                if wait>=mc["patience"]: print(f"{name}: early stopping at epoch {epoch}"); break
        writer.close(); save_loss_curve(history,out/"plots"/f"{name}_loss.png")
        state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state["model_state"]); model.eval()
        preds=[]; trues=[]
        with torch.no_grad():
            for xb,yb in te:
                p=model(xb.to(device,non_blocking=True)).float().cpu().numpy(); preds.append(p); trues.append(yb.numpy())
        yp=inv_target(np.concatenate(preds),scaler,ot_idx); yt=inv_target(np.concatenate(trues),scaler,ot_idx)
        met=regression_metrics(yt,yp); met["Model"]=name; met["BestEpoch"]=best_epoch; met["BestValLoss"]=best
        rows.append(met); json_runs[name]=met; save_forecast(yt,yp,out/"plots"/f"{name}_forecast.png",f"{name}: 24h OT forecast")
    import pandas as pd
    cols=["Model","MAE","MSE","MeanTempError","MaxTempError","MaxRiseRateError","PeakError","BestEpoch","BestValLoss"]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(out/"metrics.csv",index=False)
    with open(out/"metrics.json","w",encoding="utf-8") as f: json.dump(json_runs,f,indent=2,ensure_ascii=False)
    with open(out/"run_config.json","w",encoding="utf-8") as f: json.dump(cfg,f,indent=2,ensure_ascii=False)
    pd.DataFrame(rows).reindex(columns=cols).to_excel(out/"metrics.xlsx",index=False)
    print("\n=== FINAL COMPARISON ==="); print(pd.DataFrame(rows).reindex(columns=cols).to_string(index=False))
    print(f"\nSaved to: {out}")

if __name__=="__main__": main()
