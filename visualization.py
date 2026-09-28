from pathlib import Path
import matplotlib.pyplot as plt
def save_loss_curve(history,out):
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(8,5)); plt.plot(history["train"],label="train"); plt.plot(history["val"],label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Training / Validation Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out,dpi=160); plt.close()
def save_forecast(ytrue,ypred,out,title):
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(10,5)); plt.plot(ytrue[0],label="Actual"); plt.plot(ypred[0],label="Forecast")
    plt.xlabel("Future hour"); plt.ylabel("OT"); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out,dpi=160); plt.close()
