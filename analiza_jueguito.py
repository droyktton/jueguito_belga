"""Análisis del jueguito belga: pieza que baja por un palito dando volteretas.

1) separa las 3 realizaciones del video (la mano que coloca la pieza las delimita),
2) detecta cada voltereta (saltito) y su sentido de giro,
3) grafica el ángulo acumulado vs número de salto.

Convención: la pieza voltea hacia la izquierda del palito -> giro antihorario (+180°),
hacia la derecha -> horario (-180°), visto desde la cámara.

Uso: python3 analiza_jueguito.py [video.mp4]
"""
import sys
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "20261007_102608.mp4"
SCALE = 0.5            # se trabaja a media resolución (540x960)
DIFF_TH = 50           # umbral de diferencia entre frames consecutivos
STRIP = 80             # semiancho (px) de la franja alrededor del palito
CORE = 14              # semiancho del palito: el movimiento ahí no cuenta como voltereta
EV_MIN = 700           # px fuera del palito para considerar que hay voltereta
HAND_TH = 1000         # px de piel/buzo en la franja => hay una mano
# correcciones manuales opcionales: {(realizacion, n_salto): +1/-1}
OVERRIDE = {}


def load_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA))
    return np.array(frames), fps


def stick_line(f):
    """Recta x = a*y + b del palito (madera anaranjada saturada)."""
    h = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = (h[..., 0] > 5) & (h[..., 0] < 22) & (h[..., 1] > 110) & (h[..., 2] > 80)
    x0 = m[50:700].mean(0).argmax()
    ys, xs = [], []
    for y in range(40, 780, 10):
        lo = max(0, x0 - 60)
        row = m[y, lo:x0 + 60].astype(float)
        if row.sum() >= 4:
            ys.append(y)
            xs.append(lo + np.average(np.arange(len(row)), weights=row))
    if len(ys) < 5:
        return None
    return np.polyfit(ys, xs, 1)


def skin_mask(f):
    """Mano / buzo rojo (rojo más oscuro que la madera del palito)."""
    h = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    return ((h[..., 0] <= 5) | (h[..., 0] >= 165)) & (h[..., 1] > 70) & (h[..., 2] > 40)


def analyze(F):
    n, H, W = F.shape[:3]
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in F]
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    yy, xx = np.mgrid[0:H, 0:W]
    lines, hand, npix, xside, ycen, masks = [], [], [], [], [], []
    last = np.array([0.0, W / 2])
    skin_prev = skin_mask(F[0])
    for i in range(n):
        p = stick_line(F[i])
        p = last if p is None else p
        last = p
        lines.append(p)
        xr = xx - (p[0] * yy + p[1])
        band = (np.abs(xr) < STRIP) & (yy < 820)
        skin = skin_mask(F[i])
        hand.append((skin & band).sum())
        if i == 0:
            npix.append(0); xside.append(0.0); ycen.append(np.nan)
            masks.append(np.packbits(np.zeros((H, W), bool), axis=1))
            skin_prev = skin
            continue
        # compensar el temblor de la cámara antes de restar frames
        (dx, dy), _ = cv2.phaseCorrelate(gray[i - 1], gray[i], win)
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        cur = cv2.warpAffine(F[i], M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        d = np.abs(cur.astype(np.int16) - F[i - 1].astype(np.int16)).max(-1)
        off = (d > DIFF_TH) & band & (np.abs(xr) > CORE)
        if max(hand[-1], (skin_prev & band).sum()) > HAND_TH:
            # sólo con la mano en cuadro: descartar sus píxeles (la pieza también es rojiza)
            off &= ~cv2.dilate((skin | skin_prev).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        k = off.sum()
        masks.append(np.packbits(off, axis=1))
        npix.append(k)
        xside.append(xr[off].mean() if k else 0.0)
        ycen.append(yy[off].mean() if k else np.nan)
        skin_prev = skin
    return dict(lines=np.array(lines), hand=np.array(hand), npix=np.array(npix),
                xside=np.array(xside), ycen=np.array(ycen), masks=masks)


def segment_realizations(hand, nmin=60):
    """Tramos sin mano de al menos nmin frames. Arranca cuando la mano suelta la pieza."""
    free = hand < HAND_TH
    segs, i, n = [], 0, len(hand)
    while i < n:
        if free[i]:
            j = i
            while j < n and free[j]:
                j += 1
            if j - i >= nmin:
                segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def detect_jumps(r, a, b):
    """Volteretas: rachas de frames con mucho movimiento fuera del palito.
    Se mira desde unos frames antes de 'a' porque la primera voltereta empieza
    mientras los dedos todavía están saliendo del cuadro."""
    lo = max(1, a - 6)
    active = r["npix"][lo:b + 1] > EV_MIN
    jumps, i = [], 0
    idx = np.arange(lo, b + 1)
    while i < len(active):
        if active[i]:
            j = i
            while j + 1 < len(active) and (active[j + 1] or (j + 2 < len(active) and active[j + 2])):
                j += 1
            fr = idx[i:j + 1]
            w = r["npix"][fr].astype(float)
            xs = r["xside"][fr]
            mean_x = np.average(xs, weights=w)
            # confianza: fracción del "peso" del lado elegido
            same = w[np.sign(xs) == np.sign(mean_x)].sum() / w.sum()
            jumps.append(dict(f0=int(fr[0]), f1=int(fr[-1]), x=float(mean_x),
                              y=float(np.nanmean(r["ycen"][fr])),
                              sense=1 if mean_x < 0 else -1, conf=float(same)))
            i = j + 1
            if jumps[-1]["f0"] >= b - 3:   # es la mano que vuelve a entrar, no un salto
                jumps.pop()
        else:
            i += 1
    return jumps


def main():
    F, fps = load_frames(VIDEO)
    print(f"{len(F)} frames, {fps:.1f} fps")
    r = analyze(F)
    segs = segment_realizations(r["hand"])
    print("realizaciones (frames):", segs)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for k, (a, b) in enumerate(segs, 1):
        t0, t1 = max(0, a - 6) / fps, (b + 1) / fps
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
                        "-i", VIDEO, "-c:v", "libx264", "-crf", "20", "-an",
                        f"realizacion_{k}.mp4"], check=False)
        jumps = detect_jumps(r, a, b)
        ang = [0]
        with open(f"serie_realizacion_{k}.csv", "w") as fh:
            fh.write("n_salto,frame_ini,frame_fin,t_s,lado,sentido,angulo_acum_deg,confianza\n")
            fh.write(f"0,{a},{a},{a / fps:.3f},,0,0,\n")
            for n, J in enumerate(jumps, 1):
                s = OVERRIDE.get((k, n), J["sense"])
                ang.append(ang[-1] + 180 * s)
                fh.write(f"{n},{J['f0']},{J['f1']},{J['f0'] / fps:.3f},"
                         f"{'izq' if J['x'] < 0 else 'der'},{s:+d},{ang[-1]},{J['conf']:.2f}\n")
        seq = "".join("↺" if np.sign(d) > 0 else "↻" for d in np.diff(ang))
        print(f"R{k}: {len(jumps)} saltos  {seq}  ángulo final {ang[-1]}°")
        for n, J in enumerate(jumps, 1):
            flag = "" if J["conf"] > 0.8 else "  <-- revisar"
            print(f"   salto {n:2d}: frames {J['f0']}-{J['f1']}  x={J['x']:+.1f}  conf={J['conf']:.2f}{flag}")
        ax.step(range(len(ang)), ang, where="post", marker="o", ms=4, color=colors[(k - 1) % 3],
                label=f"realización {k}")
        save_check(F, r, jumps, k)
        save_overlay(F, r, jumps, a, b, k, fps)

    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("número de salto")
    ax.set_ylabel("ángulo acumulado [°]  (+ = antihorario)")
    lo, hi = ax.get_ylim()
    ax.set_yticks(np.arange(180 * np.floor(lo / 180), hi + 1, 360 if hi - lo > 1500 else 180))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("angulo_vs_salto.png", dpi=150)
    print("-> angulo_vs_salto.png")


def save_overlay(F, r, jumps, a, b, k, fps):
    """Video de control: palito (línea verde), píxeles de la pieza en movimiento
    (rojo = cuenta como voltereta), número de salto y sentido detectado."""
    H, W = F.shape[1:3]
    lo = max(1, a - 6)
    tmp = f"segmentacion_realizacion_{k}.avi"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), fps / 2, (W, H))  # a media velocidad
    for i in range(lo, b + 1):
        img = F[i].copy()
        m = np.unpackbits(r["masks"][i], axis=1)[:, :W].astype(bool)
        J = next(((n, J) for n, J in enumerate(jumps, 1) if J["f0"] <= i <= J["f1"]), None)
        img[m] = (0, 0, 255) if J else (0, 200, 255)
        a_, b_ = r["lines"][i]
        cv2.line(img, (int(b_ + a_ * 40), 40), (int(b_ + a_ * 820), 820), (0, 255, 0), 1)
        for s_ in (-1, 1):
            for off in (CORE, STRIP):
                cv2.line(img, (int(b_ + a_ * 40 + s_ * off), 40), (int(b_ + a_ * 820 + s_ * off), 820),
                         (120, 120, 120), 1)
        n_prev = sum(1 for J2 in jumps if J2["f1"] < i)
        cv2.putText(img, f"R{k} frame {i}  saltos: {n_prev}", (10, 30), 0, 0.7, (0, 0, 0), 2)
        if J:
            n, J = J
            txt = f"salto {n}: {'izq -> CCW +180' if J['sense'] > 0 else 'der -> CW -180'}"
            cv2.putText(img, txt, (10, 60), 0, 0.7, (0, 0, 255), 2)
        vw.write(img)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", f"segmentacion_realizacion_{k}.mp4"], check=False)
    subprocess.run(["rm", "-f", tmp])


def save_check(F, r, jumps, k):
    """Tira de verificación: para cada salto, el frame de máxima apertura con el sentido."""
    tiles = []
    for n, J in enumerate(jumps, 1):
        fr = np.arange(J["f0"], J["f1"] + 1)
        i = fr[np.argmax(r["npix"][fr])]
        a, b = r["lines"][i]
        yc = int(np.clip(J["y"], 100, F.shape[1] - 100))
        xc = int(a * yc + b)
        x0 = int(np.clip(xc - 70, 0, F.shape[2] - 140))
        tile = F[i, yc - 100:yc + 100, x0:x0 + 140].copy()
        tile = cv2.resize(tile, (280, 400))
        col = (0, 160, 0) if J["sense"] > 0 else (0, 0, 220)
        cv2.putText(tile, f"{n} f{i}", (5, 22), 0, 0.7, (0, 0, 0), 2)
        cv2.putText(tile, "CCW" if J["sense"] > 0 else "CW", (5, 390), 0, 0.9, col, 2)
        tiles.append(tile)
    if tiles:
        rows = [np.hstack(tiles[j:j + 8] + [np.full_like(tiles[0], 255)] * (8 - len(tiles[j:j + 8])))
                for j in range(0, len(tiles), 8)]
        cv2.imwrite(f"check_realizacion_{k}.jpg", np.vstack(rows))


if __name__ == "__main__":
    main()
