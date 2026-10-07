"""GIF con varias realizaciones de muchas.mp4 lado a lado, sincronizadas desde que se suelta
la pieza, con nº de salto, ángulo acumulado y CW/CCW (de <salida>/saltos.csv).

Uso: python3 hace_gif_muchas.py video.mp4 [carpeta_salida] [realizaciones separadas por coma]
"""
import os
import sys
import csv
import subprocess
import numpy as np
import cv2

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/muchas.mp4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "muchas"
REALS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "16,12,1,2,19,21").split(",")]
SCALE = 0.5
HALF_W = 70             # semiancho del recorte alrededor del palito (px a media resolución)
Y0, Y1 = 0, 620         # alto del recorte (incluye la base)
OUT_W = 110             # ancho de cada panel en el gif
FPS_OUT = 15            # media velocidad
PRE = 8                 # frames antes del inicio (la mano soltando)
POST = 15               # frames después del último salto


def main():
    c = np.load(os.path.join(OUT, "cache.npz"))
    lines = c["lines"]
    rows = list(csv.DictReader(open(os.path.join(OUT, "saltos.csv"))))
    reals = list(csv.DictReader(open(os.path.join(OUT, "realizaciones.csv"))))
    info = {}
    for k in REALS:
        rr = next(r for r in reals if int(r["realizacion"]) == k)
        jumps = [(int(r["frame_ini"]), int(r["frame_fin"]), int(r["sentido"]), int(r["angulo_acum_deg"]))
                 for r in rows if int(r["realizacion"]) == k]
        a = int(rr["frame_ini"]) - PRE
        b = jumps[-1][1] + POST
        info[k] = dict(a=a, b=b, jumps=jumps)
    want = {i for v in info.values() for i in range(v["a"], v["b"] + 1)}
    crops = {}
    cap = cv2.VideoCapture(VIDEO)     # lectura secuencial: el mp4 tiene frame rate variable
    W, H = int(cap.get(3) * SCALE), int(cap.get(4) * SCALE)
    out_h = int(OUT_W * (Y1 - Y0) / (2 * HALF_W))
    for i in range(max(want) + 1):
        if i not in want:
            cap.grab()
            continue
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        xc = int(lines[i][1] + lines[i][0] * (Y0 + Y1) / 2)
        x0 = int(np.clip(xc - HALF_W, 0, W - 2 * HALF_W))
        crops[i] = cv2.resize(f[Y0:Y1, x0:x0 + 2 * HALF_W], (OUT_W, out_h), interpolation=cv2.INTER_AREA)

    nmax = max(v["b"] - v["a"] for v in info.values())
    tmp = os.path.join(OUT, "_gif_tmp.avi")
    vw = None
    for t in range(nmax + 1):
        tiles = []
        for k, v in info.items():
            i = min(v["a"] + t, v["b"])
            img = crops[i].copy()
            done = [J for J in v["jumps"] if J[1] <= i]
            act = next((J[2] for J in v["jumps"] if J[0] <= i <= J[1]), 0)
            ang = done[-1][3] if done else 0
            cv2.rectangle(img, (0, 0), (OUT_W, 44), (255, 255, 255), -1)
            cv2.putText(img, f"R{k} n={len(done)}", (3, 16), 0, 0.45, (0, 0, 0), 1)
            col = (0, 0, 0) if act == 0 else ((0, 140, 0) if act > 0 else (0, 0, 210))
            cv2.putText(img, f"{ang:+d}", (3, 38), 0, 0.6, col, 2)
            if act:
                cv2.putText(img, "CCW" if act > 0 else "CW", (OUT_W - 40, 38), 0, 0.45, col, 2)
            tiles.append(img)
        sep = np.full((tiles[0].shape[0], 3, 3), 255, np.uint8)
        frame = tiles[0]
        for tl in tiles[1:]:
            frame = np.hstack([frame, sep, tl])
        if vw is None:
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), FPS_OUT, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    vw.release()
    pal = ("fps=%d,split[a][b];[a]palettegen=max_colors=64[p];"
           "[b][p]paletteuse=dither=bayer:bayer_scale=4" % FPS_OUT)
    dst = os.path.join(OUT, "realizaciones.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-filter_complex", pal,
                    "-loop", "0", dst], check=True)
    os.remove(tmp)
    print("->", dst)


if __name__ == "__main__":
    main()
