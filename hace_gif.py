"""GIF con las tres realizaciones lado a lado, sincronizadas desde que se suelta la pieza,
con el contador de saltos y el ángulo acumulado (leídos de serie_realizacion_k.csv).

Uso: python3 hace_gif.py [video.mp4]   (correr antes analiza_jueguito.py)
"""
import sys
import csv
import subprocess
import numpy as np
import cv2
from analiza_jueguito import load_frames, stick_line

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "20261007_102608.mp4"
HALF_W = 90            # semiancho del recorte alrededor del palito (px a media resolución)
Y0, Y1 = 0, 830        # alto del recorte
OUT_W = 150            # ancho de cada panel en el gif
FPS_OUT = 15           # el gif va a media velocidad
PRE = 6                # frames antes del inicio de la realización (la mano soltando)


def read_series(k):
    with open(f"serie_realizacion_{k}.csv") as fh:
        rows = list(csv.DictReader(fh))
    start = int(rows[0]["frame_ini"])
    end = int(rows[-1]["frame_fin"]) + 25   # un poco después del último salto
    jumps = [(int(r["frame_ini"]), int(r["frame_fin"]), int(r["sentido"]), int(r["angulo_acum_deg"]))
             for r in rows[1:]]
    return start, end, jumps


def panel(f, k, n, ang, active):
    p = stick_line(f)
    xc = int(p[1] + p[0] * (Y0 + Y1) / 2) if p is not None else f.shape[1] // 2
    x0 = int(np.clip(xc - HALF_W, 0, f.shape[1] - 2 * HALF_W))
    c = f[Y0:Y1, x0:x0 + 2 * HALF_W]
    c = cv2.resize(c, (OUT_W, int(OUT_W * (Y1 - Y0) / (2 * HALF_W))), interpolation=cv2.INTER_AREA)
    cv2.rectangle(c, (0, 0), (OUT_W, 60), (255, 255, 255), -1)
    cv2.putText(c, f"R{k}  salto {n}", (4, 20), 0, 0.5, (0, 0, 0), 2)
    col = (0, 0, 0) if active == 0 else ((0, 140, 0) if active > 0 else (0, 0, 210))
    cv2.putText(c, f"{ang:+d} deg", (4, 48), 0, 0.6, col, 2)
    if active:
        cv2.putText(c, "CCW" if active > 0 else "CW", (OUT_W - 45, 48), 0, 0.5, col, 2)
    return c


def main():
    F, fps = load_frames(VIDEO)
    series = [read_series(k) for k in (1, 2, 3)]
    nmax = max(e - (s - PRE) for s, e, _ in series)
    tmp = "_gif_tmp.avi"
    vw = None
    for t in range(nmax + 1):
        tiles = []
        for k, (s, e, jumps) in enumerate(series, 1):
            i = min(s - PRE + t, e, len(F) - 1)
            done = [J for J in jumps if J[1] <= i]
            act = next((J[2] for J in jumps if J[0] <= i <= J[1]), 0)
            tiles.append(panel(F[i], k, len(done), done[-1][3] if done else 0, act))
        sep = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
        img = np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]])
        if vw is None:
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), FPS_OUT,
                                 (img.shape[1], img.shape[0]))
        vw.write(img)
    vw.release()
    pal = "fps=%d,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" % FPS_OUT
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-filter_complex", pal,
                    "-loop", "0", "realizaciones.gif"], check=True)
    subprocess.run(["rm", "-f", tmp])
    print("-> realizaciones.gif")


if __name__ == "__main__":
    main()
