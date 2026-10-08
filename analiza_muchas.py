"""Análisis por lotes del jueguito belga (video largo con muchas realizaciones).

Toma con fondo blanco, cámara (casi) fija y palito de madera clara.

Pasada 1 (lenta, se guarda en <salida>/cache.npz): recorre el video frame a frame y guarda
  - recta del palito, cantidad de "mano" en cuadro,
  - píxeles que cambiaron respecto del frame anterior a cada lado del palito
    (máscara recortada en una franja alrededor del palito, para los videos de control).
Pasada 2: separa realizaciones, detecta saltos y su sentido, escribe CSV, gráficos,
  clips por realización y videos de control.

Convención: la pieza voltea hacia la izquierda del palito -> antihorario (+180°),
hacia la derecha -> horario (-180°), visto desde la cámara.

Uso: python3 analiza_muchas.py video.mp4 [carpeta_salida] [--sin-videos] [--sin-primer-salto]
"""
import os
import sys
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/muchas.mp4"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(VIDEO))[0]
SCALE = 0.5            # se trabaja a media resolución (540x960)
Y_TOP, Y_BOT = 20, 600 # zona útil del palito (arriba de la base de madera)
DIFF_TH = 40           # umbral de diferencia entre frames consecutivos
STRIP = 80             # semiancho (px) de la franja alrededor del palito
Y_FIT = 480            # el palito se ajusta sólo por encima de esta altura
STICK_W = 13           # ancho del palito (px)
CORE = 12              # semiancho del palito: el movimiento ahí no cuenta como voltereta
EV_MIN = 250           # px fuera del palito para considerar que hay voltereta
PIECE_MAX = 2500       # px: una mancha de movimiento más grande no es la pieza
HAND_TH = 4000         # px de piel en cuadro => hay una mano
MIN_LEN = 45           # frames mínimos de una realización
OVERRIDE = {}          # correcciones manuales: {(realizacion, n_salto): +1/-1}


def stick_line(f):
    """Recta x = a*y + b del palito: madera clara (algo saturada) sobre pared blanca."""
    h = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = (h[..., 0] >= 6) & (h[..., 0] <= 24) & (h[..., 1] > 28) & (h[..., 1] < 110) & (h[..., 2] > 140)
    m[:Y_TOP] = False
    m[Y_FIT:] = False
    prof = m.mean(0)
    x0 = int(prof.argmax())
    if prof[x0] < 0.3:
        return None
    # El palito es la primera racha de madera desde la izquierda: a su derecha suele haber
    # un reflejo anaranjado sobre la pared que se pega a la máscara y la desplazaría.
    ys, xs = [], []
    for y in range(Y_TOP + 10, Y_FIT, 10):    # cerca de la base hay brillo anaranjado
        lo = max(0, x0 - 50)
        row = m[y, lo:x0 + 50]
        idx = np.nonzero(row)[0]
        if len(idx) < 5:
            continue
        brk = np.nonzero(np.diff(idx) > 1)[0]
        ends = np.r_[brk, len(idx) - 1]
        starts = np.r_[0, brk + 1]
        for a_, b_ in zip(starts, ends):
            if idx[b_] - idx[a_] + 1 >= 5:
                ys.append(y)
                xs.append(lo + idx[a_] + min(idx[b_] - idx[a_] + 1, STICK_W) / 2)
                break
    if len(ys) < 8:
        return None
    p = np.polyfit(ys, xs, 1)
    res = np.abs(np.polyval(p, ys) - xs)
    keep = res < 4                       # descartar filas donde la pieza o la mano tapan
    if keep.sum() >= 8:
        p = np.polyfit(np.array(ys)[keep], np.array(xs)[keep], 1)
    return p


def skin_mask(f):
    h = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    # piel: bastante más saturada y oscura que la madera y el brillo amarillo junto a la base
    return (((h[..., 0] <= 14) | (h[..., 0] >= 165)) & (h[..., 1] > 110) & (h[..., 2] > 30) & (h[..., 2] < 185))


def pass1(path, cache):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nfr = min(nfr, int(os.environ.get("JB_MAXF", nfr)))   # para pruebas rápidas
    H, W = int(cap.get(4) * SCALE), int(cap.get(3) * SCALE)
    yy, xx = np.mgrid[0:H, 0:W]
    lines = np.zeros((nfr, 2)); hand = np.zeros(nfr, int); npix = np.zeros(nfr, int)
    xside = np.zeros(nfr); ycen = np.full(nfr, np.nan); x0s = np.zeros(nfr, int)
    masks = np.zeros((nfr, H, (2 * STRIP) // 8), np.uint8)
    win = None
    prev = prev_g = None
    last = np.array([0.0, W / 2])
    i = 0
    while True:
        ok, f = cap.read()
        if not ok or i >= nfr:
            break
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        p = stick_line(f)
        p = last if p is None else p
        last = p
        lines[i] = p
        xr = xx - (p[0] * yy + p[1])
        sk = skin_mask(f)
        sk[Y_BOT:] = False
        hand[i] = (sk & (np.abs(xr) > CORE)).sum()
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (W // 2, H // 2)).astype(np.float32)
        if prev is not None:
            if win is None:
                win = cv2.createHanningWindow((W // 2, H // 2), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(prev_g, g, win)
            M = np.float32([[1, 0, -2 * dx], [0, 1, -2 * dy]])
            cur = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            d = np.abs(cur.astype(np.int16) - prev.astype(np.int16)).max(-1)
            # descartar manchas de movimiento grandes (mano y su sombra); la pieza es chica
            nl, lab, st, _ = cv2.connectedComponentsWithStats((d > DIFF_TH).astype(np.uint8), connectivity=8)
            big = np.nonzero(st[:, cv2.CC_STAT_AREA] > PIECE_MAX)[0]
            big = big[big > 0]
            if len(big):
                d = d.copy()
                d[np.isin(lab, big)] = 0
            band = (np.abs(xr) < STRIP) & (yy > Y_TOP) & (yy < Y_BOT)
            off = (d > DIFF_TH) & band & (np.abs(xr) > CORE)
            k = off.sum()
            npix[i] = k
            if k:
                xside[i] = xr[off].mean()
                ycen[i] = yy[off].mean()
            x0 = int(np.clip(round(p[1] + p[0] * H / 2) - STRIP, 0, W - 2 * STRIP))
            x0s[i] = x0
            masks[i] = np.packbits((d > DIFF_TH)[:, x0:x0 + 2 * STRIP], axis=1)
        prev, prev_g = f, g
        i += 1
        if i % 500 == 0:
            print(f"  {i}/{nfr}", flush=True)
    n = i
    np.savez_compressed(cache, fps=fps, lines=lines[:n], hand=hand[:n], npix=npix[:n],
                        xside=xside[:n], ycen=ycen[:n], x0s=x0s[:n], masks=masks[:n])


# ---------------------------------------------------------------- pasada 2

EV_LO = 200            # px: umbral para que un frame forme parte de un salto
EV_PEAK = 500          # px: un salto tiene que llegar al menos a esto en algún frame
MAX_DUR = 9            # frames: un salto más largo se parte en dos
Y_LAND = 565           # px (media res.): más abajo es la pieza cayendo sobre la base
MIN_JUMPS = 3          # realizaciones con menos saltos se descartan (mano que pasa, etc.)


def segment_realizations(hand):
    """Tramos sin mano de al menos MIN_LEN frames."""
    free = hand < HAND_TH
    segs, i, n = [], 0, len(hand)
    while i < n:
        if free[i]:
            j = i
            while j < n and free[j]:
                j += 1
            if j - i >= MIN_LEN:
                segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def detect_jumps(c, a, b):
    """Saltos: rachas de frames con movimiento fuera del palito (se tolera 1 frame de hueco).
    Sentido: lado (izq/der del palito) hacia el que se abre la pieza, pesado por píxeles."""
    npix, xs_all, yc = c["npix"], c["xside"], c["ycen"]
    act = npix[a:b + 1] > EV_LO
    jumps, i, n = [], 0, len(act)
    while i < n:
        if not act[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (act[j + 1] or (j + 2 < n and act[j + 2])):
            j += 1
        fr_all = np.arange(a + i, a + j + 1)
        i = j + 1
        # un evento demasiado largo son dos saltos seguidos: cortar en el mínimo de movimiento
        pieces = [fr_all]
        if len(fr_all) > MAX_DUR:
            mid = fr_all[2:-2]
            cut = mid[np.argmin(npix[mid])]
            pieces = [fr_all[fr_all < cut], fr_all[fr_all > cut]]
        for fr in pieces:
            fr = fr[npix[fr] > EV_LO]
            if len(fr) == 0 or npix[fr].max() < EV_PEAK:
                continue
            J = _jump(fr, npix, xs_all, yc)
            # la pieza siempre baja: un "salto" más arriba que el anterior es la mano
            # que vuelve a entrar; y por debajo de Y_LAND es la pieza cayendo sobre la base
            if jumps and J["y"] < jumps[-1]["y"] - 30:
                continue
            if J["y"] > Y_LAND:
                continue
            jumps.append(J)
    return jumps


def _jump(fr, npix, xs_all, yc):
    w = npix[fr].astype(float)
    xs = xs_all[fr]
    mx = np.average(xs, weights=w)
    conf = w[np.sign(xs) == np.sign(mx)].sum() / w.sum()
    return dict(f0=int(fr[0]), f1=int(fr[-1]), x=float(mx), y=float(np.average(yc[fr], weights=w)),
                sense=1 if mx < 0 else -1, conf=float(conf))


def read_frames(path, frames_wanted):
    """Generador (i, frame a media resolución) sólo para los frames pedidos, en orden."""
    want = sorted(set(frames_wanted))
    cap = cv2.VideoCapture(path)
    W, H = int(cap.get(3) * SCALE), int(cap.get(4) * SCALE)
    k, i = 0, 0
    while k < len(want):
        if i < want[k]:
            cap.grab()
            i += 1
            continue
        ok, f = cap.read()
        if not ok:
            break
        yield i, cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        i += 1
        k += 1


def pass2(c, real, suf="", titulo=""):
    fps = float(c["fps"])
    # ---- tablas
    with open(os.path.join(OUT, f"saltos{suf}.csv"), "w") as fh:
        fh.write("realizacion,n_salto,frame_ini,frame_fin,t_s,lado,sentido,angulo_acum_deg,altura_px,confianza\n")
        for r in real:
            for n, J in enumerate(r["jumps"], 1):
                fh.write(f"{r['k']},{n},{J['f0']},{J['f1']},{(J['f0'] - r['a']) / fps:.3f},"
                         f"{'izq' if J['sense'] > 0 else 'der'},{J['s']:+d},{r['ang'][n]},"
                         f"{J['y'] / SCALE:.0f},{J['conf']:.2f}\n")
    with open(os.path.join(OUT, f"realizaciones{suf}.csv"), "w") as fh:
        fh.write("realizacion,frame_ini,frame_fin,t_ini_s,n_saltos,n_ccw,n_cw,angulo_final_deg,secuencia\n")
        for r in real:
            s_ = np.array([J["s"] for J in r["jumps"]])
            fh.write(f"{r['k']},{r['a']},{r['b']},{r['a'] / fps:.2f},{len(s_)},{(s_ > 0).sum()},{(s_ < 0).sum()},"
                     f"{r['ang'][-1]},{''.join('L' if v > 0 else 'R' for v in s_)}\n")

    # ---- caminatas
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for idx, r in enumerate(real):
        ax.plot(range(len(r["ang"])), np.array(r["ang"]) / 180, "-", lw=1, alpha=0.6,
                color=cmap(idx / max(1, len(real) - 1)))
    nmax = max(len(r["ang"]) for r in real)
    M = np.full((len(real), nmax), np.nan)
    for idx, r in enumerate(real):
        M[idx, :len(r["ang"])] = np.array(r["ang"]) / 180
    cnt = (~np.isnan(M)).sum(0)
    ok = cnt >= 3
    mean = np.nanmean(M, 0)
    ax.plot(np.arange(nmax)[ok], mean[ok], "k-", lw=2.5, label="media")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("número de salto n")
    ax.set_ylabel("ángulo acumulado θ / 180°   (+ = antihorario)")
    ax.set_title(f"{len(real)} realizaciones{titulo}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"angulo_vs_salto{suf}.png"), dpi=150)
    plt.close(fig)

    # ---- estadística
    S = [np.array([J["s"] for J in r["jumps"]]) for r in real]
    allS = np.concatenate(S)
    nn = np.arange(nmax)
    msd = np.nanmean(M ** 2, 0)
    pairs = np.concatenate([s_[1:] * s_[:-1] for s_ in S if len(s_) > 1])
    lags = np.arange(1, 8)
    corr = [np.mean(np.concatenate([s_[l:] * s_[:-l] for s_ in S if len(s_) > l]))
            if any(len(s_) > l for s_ in S) else np.nan for l in lags]
    pos = np.arange(1, nmax)
    p_ccw = [np.mean([s_[k - 1] > 0 for s_ in S if len(s_) >= k]) for k in pos]
    n_at = [sum(len(s_) >= k for s_ in S) for k in pos]
    finals = np.array([r["ang"][-1] / 180 for r in real])
    nj = np.array([len(s_) for s_ in S])

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    a = axs[0, 0]
    a.plot(nn[ok], msd[ok], "o-", label="⟨θ²⟩ medido")
    a.plot(nn[ok], nn[ok], "k--", label="caminata al azar sin memoria (= n)")
    a.plot(nn[ok], mean[ok] ** 2, ":", color="gray", label="⟨θ⟩²")
    a.set_xlabel("n"); a.set_ylabel("⟨θ²⟩ / 180°²"); a.legend(fontsize=8); a.grid(alpha=0.3)
    a.set_title("desplazamiento cuadrático medio")
    a = axs[0, 1]
    a.bar(lags, corr)
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("separación ℓ (saltos)"); a.set_ylabel("⟨s_n s_{n+ℓ}⟩")
    a.set_title(f"autocorrelación del sentido (P(repetir) = {(pairs > 0).mean():.2f})")
    a.grid(alpha=0.3)
    a = axs[0, 2]
    a.plot(pos, p_ccw, "o-")
    for k_, (pp, m_) in enumerate(zip(p_ccw, n_at)):
        if m_ < 3:
            break
    a.axhline(0.5, color="k", lw=0.6)
    a.set_ylim(0, 1)
    a.set_xlabel("salto n"); a.set_ylabel("P(antihorario)")
    a.set_title(f"P(antihorario) global = {(allS > 0).mean():.2f}  ({len(allS)} saltos)")
    a.grid(alpha=0.3)
    a = axs[1, 0]
    bins = np.arange(finals.min() - 1.5, finals.max() + 2.5, 2) if len(finals) else 10
    a.hist(finals, bins=np.arange(np.floor(finals.min()) - 0.5, np.ceil(finals.max()) + 1.5, 1), rwidth=0.85,
           label="medido")
    # esperado con giros de ±180° al azar e independientes: suma de binomiales, una por
    # realización con su propio número de saltos (respeta la paridad de θ_f/180 con n)
    from math import comb
    xm = np.arange(-nj.max(), nj.max() + 1)
    expct = np.zeros(len(xm))
    for n_ in nj:
        for k_ in range(n_ + 1):
            expct[2 * k_ - n_ + nj.max()] += comb(n_, k_) / 2 ** n_
    # banda: intervalo central del 68 % de los conteos por bin en repeticiones simuladas de
    # todas las realizaciones (asimétrica, entera y nunca negativa)
    rng = np.random.default_rng(0)
    sims = 2 * rng.binomial(nj[None, :], 0.5, size=(20000, len(nj))) - nj[None, :]
    cnts = np.stack([(sims == v).sum(1) for v in xm], 1)
    lo, hi = np.percentile(cnts, [16, 84], axis=0)
    keep = expct > 1e-3
    # el intervalo no siempre contiene a la media (p. ej. media 0.2 -> [0, 0]): se dibuja aparte
    a.vlines(xm[keep], lo[keep], hi[keep], color="k", lw=1.5, label="banda 68 % (simulada)")
    a.plot(xm[keep], expct[keep], "o", color="k", ms=4, label="esperado: ±180° al azar")
    a.legend(fontsize=7)
    a.set_xlabel("ángulo final / 180°"); a.set_ylabel("realizaciones")
    a.set_title(f"ángulo final: media {finals.mean():+.2f}, desv. {finals.std():.2f}")
    a = axs[1, 1]
    a.hist(nj, bins=np.arange(nj.min() - 0.5, nj.max() + 1.5, 1), rwidth=0.85)
    a.set_xlabel("saltos por realización"); a.set_ylabel("realizaciones")
    a.set_title(f"saltos por realización: media {nj.mean():.1f}")
    fig.suptitle(titulo.strip(" ,"), fontsize=11)
    a = axs[1, 2]
    # largo de rachas de saltos en el mismo sentido
    runs = []
    for s_ in S:
        if len(s_) == 0:
            continue
        cur = 1
        for k_ in range(1, len(s_)):
            if s_[k_] == s_[k_ - 1]:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
    runs = np.array(runs)
    hb = np.arange(0.5, runs.max() + 1.5, 1)
    a.hist(runs, bins=hb, rwidth=0.85, density=True, label="medido")
    q = (pairs > 0).mean()
    kk = np.arange(1, runs.max() + 1)
    a.plot(kk, 0.5 ** kk, "k--", label="sin memoria (2⁻ᵏ)")
    a.set_yscale("log")
    a.set_xlabel("largo de racha en el mismo sentido"); a.set_ylabel("frecuencia")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"estadistica{suf}.png"), dpi=130)
    plt.close(fig)
    return dict(p_ccw=(allS > 0).mean(), p_repeat=(pairs > 0).mean(), nsaltos=len(allS),
                corr1=corr[0], final_mean=finals.mean(), final_std=finals.std())


def tests(S, nsim=20000, seed=0):
    """Tests contra una caminata al azar sin sesgo ni memoria. S: lista de arrays de ±1."""
    from scipy.stats import binomtest
    rng = np.random.default_rng(seed)
    allS = np.concatenate(S)
    L = []
    k = int((allS > 0).sum())
    L.append(f"saltos: {len(allS)} en {len(S)} realizaciones")
    L.append(f"sesgo global: antihorario {k}/{len(allS)} = {k / len(allS):.2f}  "
             f"(binomial p = {binomtest(k, len(allS)).pvalue:.3f})")

    def disp(SS):
        return np.mean([x.sum() ** 2 for x in SS]) / np.mean([len(x) for x in SS])

    def rep(SS):
        return np.mean(np.concatenate([x[1:] == x[:-1] for x in SS if len(x) > 1]))

    for lab, SS in (("todas", S), ("sin la más extrema", None)):
        if SS is None:
            ext = int(np.argmax([abs(x.sum()) for x in S]))
            SS = [x for i, x in enumerate(S) if i != ext]
            lab = f"sin R{ext + 1} (|θ_f| máximo)"
        o = disp(SS)
        nul = np.array([disp([rng.choice([-1, 1], len(x)) for x in SS]) for _ in range(nsim)])
        L.append(f"dispersión ⟨θ_f²⟩/⟨n⟩ [{lab}]: {o:.2f}  (moneda justa {nul.mean():.2f} ± {nul.std():.2f}, "
                 f"p una cola = {np.mean(nul >= o):.3f})")
    o = rep(S)
    nul1 = np.array([rep([rng.choice([-1, 1], len(x)) for x in S]) for _ in range(nsim)])
    nul2 = np.array([rep([rng.permutation(x) for x in S]) for _ in range(nsim)])
    L.append(f"P(repetir sentido): {o:.3f}  vs moneda justa {nul1.mean():.3f} ± {nul1.std():.3f} "
             f"(p = {np.mean(np.abs(nul1 - nul1.mean()) >= abs(o - nul1.mean())):.3f});  "
             f"vs permutar dentro de cada realización {nul2.mean():.3f} ± {nul2.std():.3f} "
             f"(p = {np.mean(np.abs(nul2 - nul2.mean()) >= abs(o - nul2.mean())):.3f})")
    L.append("P(antihorario) por número de salto:")
    for n in range(1, max(len(x) for x in S) + 1):
        v = np.array([x[n - 1] for x in S if len(x) >= n])
        if len(v) < 5:
            break
        kk = int((v > 0).sum())
        L.append(f"   n={n:2d}: {kk:2d}/{len(v):2d} = {kk / len(v):.2f}   p = {binomtest(kk, len(v)).pvalue:.3f}")
    nmax = max(n for n in range(1, 30) if sum(len(x) >= n for x in S) >= 20)
    th = np.array([x[:nmax].sum() for x in S if len(x) >= nmax], float)
    L.append(f"⟨θ²⟩ en n={nmax} ({len(th)} realizaciones): {np.mean(th ** 2):.1f}  "
             f"(sin memoria: {nmax} ± {nmax * np.sqrt(2 / len(th)):.1f})")
    return "\n".join(L)


def make_videos(c, real):
    """Clips por realización (del video original) + video de control de la segmentación
    + tira de verificación con un recorte por salto."""
    fps = float(c["fps"])
    # clips por realización (se leen con OpenCV: el ffmpeg de snap no puede abrir /tmp)
    os.makedirs(os.path.join(OUT, "realizaciones"), exist_ok=True)
    for r in real:
        tmp = os.path.join(OUT, "_clip.avi")
        vw = None
        for i, f in read_frames(VIDEO, range(max(0, r["a"] - 15), r["b"] + 1)):
            if vw is None:
                vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), fps, (f.shape[1], f.shape[0]))
            vw.write(f)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "24",
                        "-pix_fmt", "yuv420p",
                        os.path.join(OUT, "realizaciones", f"realizacion_{r['k']:02d}.mp4")], check=False)
        os.remove(tmp)
    # video de control: todas las realizaciones seguidas, recorte alrededor del palito
    frames = [i for r in real for i in range(r["a"], r["b"] + 1)]
    owner = {i: r for r in real for i in range(r["a"], r["b"] + 1)}
    CW_ = 2 * STRIP + 80
    tmp = os.path.join(OUT, "_tmp.avi")
    vw = None
    tiles = {}
    want_tiles = {}
    for r in real:
        for n, J in enumerate(r["jumps"], 1):
            fr = np.arange(J["f0"], J["f1"] + 1)
            want_tiles[int(fr[np.argmax(c["npix"][fr])])] = (r["k"], n, J)
    for i, f in read_frames(VIDEO, frames):
        r = owner[i]
        H, W = f.shape[:2]
        x0 = int(c["x0s"][i])
        m = np.unpackbits(c["masks"][i], axis=1).astype(bool)
        J = next(((n, J) for n, J in enumerate(r["jumps"], 1) if J["f0"] <= i <= J["f1"]), None)
        img = f.copy()
        sub = img[:, x0:x0 + 2 * STRIP]
        sub[m] = (0, 0, 255) if J else (0, 200, 255)
        a_, b_ = c["lines"][i]
        cv2.line(img, (int(b_ + a_ * Y_TOP), Y_TOP), (int(b_ + a_ * Y_BOT), Y_BOT), (0, 255, 0), 1)
        xc = int(np.clip(b_ + a_ * H / 2 - CW_ / 2, 0, W - CW_))
        crop = img[:Y_BOT + 40, xc:xc + CW_].copy()
        nprev = sum(1 for J2 in r["jumps"] if J2["f1"] < i)
        ang = r["ang"][nprev]
        cv2.rectangle(crop, (0, 0), (CW_, 58), (255, 255, 255), -1)
        cv2.putText(crop, f"R{r['k']} f{i} n={nprev}", (4, 20), 0, 0.5, (0, 0, 0), 1)
        cv2.putText(crop, f"{ang:+d} deg", (4, 48), 0, 0.65, (0, 0, 0), 2)
        if J:
            n, J = J
            col = (0, 140, 0) if J["sense"] > 0 else (0, 0, 210)
            cv2.putText(crop, "CCW" if J["sense"] > 0 else "CW", (CW_ - 55, 48), 0, 0.6, col, 2)
        if vw is None:
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), fps, (crop.shape[1], crop.shape[0]))
        vw.write(crop)
        if i in want_tiles:
            k, n, J = want_tiles[i]
            yc = int(np.clip(J["y"], 60, H - 60))
            xcc = int(np.clip(a_ * yc + b_ - 60, 0, W - 120))
            t = cv2.resize(f[yc - 60:yc + 60, xcc:xcc + 120], (120, 120))
            col = (0, 140, 0) if J["s"] > 0 else (0, 0, 210)
            cv2.putText(t, f"{n}", (3, 14), 0, 0.45, (0, 0, 0), 1)
            cv2.putText(t, "L" if J["s"] > 0 else "R", (100, 115), 0, 0.5, col, 2)
            if J["conf"] < 0.8:
                cv2.rectangle(t, (0, 0), (119, 119), (0, 160, 255), 2)
            tiles.setdefault(k, []).append(t)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "26",
                    "-pix_fmt", "yuv420p", os.path.join(OUT, "segmentacion.mp4")], check=False)
    os.remove(tmp)
    # tira de verificación: una fila por realización
    ncol = max(len(v) for v in tiles.values())
    rows = []
    for k in sorted(tiles):
        lab = np.full((120, 50, 3), 255, np.uint8)
        cv2.putText(lab, f"R{k}", (4, 65), 0, 0.6, (0, 0, 0), 2)
        row = tiles[k] + [np.full((120, 120, 3), 255, np.uint8)] * (ncol - len(tiles[k]))
        rows.append(np.hstack([lab] + row))
    cv2.imwrite(os.path.join(OUT, "check_saltos.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "cache.npz")
    if not os.path.exists(cache):
        print("pasada 1:", VIDEO)
        pass1(VIDEO, cache)
    c = dict(np.load(cache))
    fps = float(c["fps"])
    segs = segment_realizations(c["hand"])
    real = []
    for a, b in segs:
        jumps = detect_jumps(c, a, b)
        if len(jumps) < MIN_JUMPS:
            print(f"  descarto tramo {a}-{b}: {len(jumps)} saltos")
            continue
        k = len(real) + 1
        ang = [0]
        for n, J in enumerate(jumps, 1):
            J["s"] = OVERRIDE.get((k, n), J["sense"])
            ang.append(ang[-1] + 180 * J["s"])
        real.append(dict(k=k, a=a, b=b, jumps=jumps, ang=ang))
        seq = "".join("L" if J["s"] > 0 else "R" for J in jumps)
        low = [n for n, J in enumerate(jumps, 1) if J["conf"] < 0.8]
        print(f"R{k:02d} frames {a:5d}-{b:5d}  {len(jumps):2d} saltos  {seq:<16s} final {ang[-1]:+5d}°"
              + (f"   revisar saltos {low}" if low else ""))
    suf, titulo = "", ""
    if "--sin-primer-salto" in sys.argv:
        # el 1er salto depende de cómo se coloca la pieza: la caminata arranca en el 2º
        for r in real:
            r["jumps"] = r["jumps"][1:]
            r["ang"] = [0] + list(np.cumsum([180 * J["s"] for J in r["jumps"]]))
        suf, titulo = "_sin1", ", sin el 1er salto"
    st = pass2(c, real, suf, titulo)
    txt = tests([np.array([J["s"] for J in r["jumps"]]) for r in real])
    print(txt)
    with open(os.path.join(OUT, f"tests{suf}.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(f"{len(real)} realizaciones, {st['nsaltos']} saltos: P(antihorario)={st['p_ccw']:.2f}, "
          f"P(repetir sentido)={st['p_repeat']:.2f}, <s_n s_n+1>={st['corr1']:+.2f}, "
          f"ángulo final {st['final_mean']:+.2f} ± {st['final_std']:.2f} (x180°)")
    if "--sin-videos" not in sys.argv and not suf:
        make_videos(c, real)
    print("salida en", OUT)
