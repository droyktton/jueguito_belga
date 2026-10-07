"""¿La inclinación de la pieza en reposo decide el sentido del salto siguiente?

Para cada salto de <salida>/saltos.csv toma los frames de reposo justo antes (f0-5 .. f0-2),
segmenta la pieza por sus partes oscuras y sus puntas blancas y ajusta x = m*y + c
(x relativo al palito, y hacia abajo). Pendiente m > 0: la pieza está inclinada como "\\"
(arriba a la izquierda, abajo a la derecha); m < 0: como "/".

Uso: python3 configuracion.py video.mp4 [carpeta_salida]   (después de analiza_muchas.py)
"""
import os
import sys
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/muchas.mp4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "muchas"
SCALE = 0.5
REST = range(-5, -1)    # frames de reposo relativos al inicio del salto
DY = 30                 # px (media res.): la pieza en reposo está algo más arriba que el centro del salto
HALF_H, HALF_W = 50, 35


def main():
    c = np.load(os.path.join(OUT, "cache.npz"))
    lines = c["lines"]
    rows = list(csv.DictReader(open(os.path.join(OUT, "saltos.csv"))))
    want = {}
    for r in rows:
        for d in REST:
            want.setdefault(int(r["frame_ini"]) + d, []).append(r)
    cap = cv2.VideoCapture(VIDEO)     # lectura secuencial: el mp4 tiene frame rate variable
    W, H = int(cap.get(3) * SCALE), int(cap.get(4) * SCALE)
    yy, xx = np.mgrid[0:H, 0:W]
    acc = {}
    for i in range(max(want) + 1):
        if i not in want:
            cap.grab()
            continue
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        xr = xx - (lines[i][0] * yy + lines[i][1])
        piece = (hsv[..., 2] < 100) | ((hsv[..., 2] > 225) & (hsv[..., 1] < 40))
        for r in want[i]:
            yc = float(r["altura_px"]) * SCALE - DY
            m = piece & (np.abs(yy - yc) < HALF_H) & (np.abs(xr) < HALF_W)
            if m.sum() < 20:
                continue
            ys, xs = yy[m].astype(float), xr[m]
            slope = np.polyfit(ys, xs, 1)[0]
            acc.setdefault((r["realizacion"], r["n_salto"]), []).append(slope)

    out = []
    for r in rows:
        v = acc.get((r["realizacion"], r["n_salto"]))
        if v:
            out.append((int(r["realizacion"]), int(r["n_salto"]), int(r["sentido"]), float(np.median(v))))
    with open(os.path.join(OUT, "inclinacion.csv"), "w") as fh:
        fh.write("realizacion,n_salto,sentido,pendiente_reposo\n")
        for k, n, s, m in out:
            fh.write(f"{k},{n},{s:+d},{m:.4f}\n")
    s = np.array([o[2] for o in out])
    m = np.array([o[3] for o in out])
    pred = np.where(m > 0, 1, -1)
    acc_ = (pred == s).mean()
    rng = np.random.default_rng(0)
    null = [(rng.permutation(pred) == s).mean() for _ in range(20000)]
    p = np.mean(np.abs(np.array(null) - 0.5) >= abs(acc_ - 0.5))
    print(f"{len(out)} saltos con inclinación medida")
    print(f"'\\' (m>0) -> antihorario, '/' (m<0) -> horario: acierta {acc_:.2f}  (azar 0.5, p={p:.4f})")
    for lab, sel in (("m > 0  (\\)", m > 0), ("m < 0  (/)", m < 0)):
        print(f"  {lab}: {sel.sum():3d} saltos, P(antihorario) = {(s[sel] > 0).mean():.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(np.percentile(m, 1), np.percentile(m, 99), 30)
    ax[0].hist(m[s > 0], bins=bins, alpha=0.6, label="salto antihorario (L)")
    ax[0].hist(m[s < 0], bins=bins, alpha=0.6, label="salto horario (R)")
    ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_xlabel("pendiente de la pieza en reposo dx/dy   (>0: '\\', <0: '/')")
    ax[0].set_ylabel("saltos")
    ax[0].legend()
    ax[0].set_title(f"acierto de la regla: {acc_:.2f}")
    q = np.quantile(m, np.linspace(0, 1, 7))
    mid, pc = [], []
    for a, b in zip(q[:-1], q[1:]):
        sel = (m >= a) & (m <= b)
        mid.append(np.median(m[sel])); pc.append((s[sel] > 0).mean())
    ax[1].plot(mid, pc, "o-")
    ax[1].axhline(0.5, color="k", lw=0.6); ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("pendiente en reposo (sextiles)")
    ax[1].set_ylabel("P(antihorario)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "inclinacion.png"), dpi=130)


if __name__ == "__main__":
    main()
