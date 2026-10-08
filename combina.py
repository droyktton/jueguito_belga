"""Junta los saltos de varios videos ya analizados con analiza_muchas.py y rehace la
estadística (mismos gráficos y tests) sobre el conjunto.

Uso: python3 combina.py carpeta_salida carpeta1[:k1,k2] carpeta2[:k3] ... [--sin-primer-salto]
     (":k1,k2" = realizaciones de esa carpeta a excluir)
Lee <carpeta>/saltos.csv (o saltos_sin1.csv con --sin-primer-salto). Las realizaciones se
renumeran 1..N; la correspondencia queda en <salida>/origen.csv.
"""
import os
import sys
import csv
import numpy as np
import analiza_muchas as A


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out, srcs = args[0], args[1:]
    sin1 = "--sin-primer-salto" in sys.argv
    suf, titulo = ("_sin1", ", sin el 1er salto") if sin1 else ("", "")
    os.makedirs(out, exist_ok=True)
    A.OUT = out
    real, origen = [], []
    for src in srcs:
        d, _, ex = src.partition(":")
        drop = {int(x) for x in ex.split(",") if x}
        fps = float(np.load(os.path.join(d, "cache.npz"))["fps"])
        rows = list(csv.DictReader(open(os.path.join(d, f"saltos{suf}.csv"))))
        reals = list(csv.DictReader(open(os.path.join(d, f"realizaciones{suf}.csv"))))
        for rr in reals:
            k0 = int(rr["realizacion"])
            if k0 in drop:
                continue
            jumps = [dict(f0=int(r["frame_ini"]), f1=int(r["frame_fin"]), y=float(r["altura_px"]) * A.SCALE,
                          conf=float(r["confianza"]), s=int(r["sentido"]), sense=int(r["sentido"]))
                     for r in rows if int(r["realizacion"]) == k0]
            if not jumps:
                continue
            ang = [0] + list(np.cumsum([180 * J["s"] for J in jumps]))
            # 'a' en unidades del fps común (t_s en saltos.csv sigue siendo relativo al inicio)
            real.append(dict(k=len(real) + 1, a=int(rr["frame_ini"]), b=int(rr["frame_fin"]), jumps=jumps,
                             ang=[int(v) for v in ang]))
            origen.append((len(real), d, k0))
    with open(os.path.join(out, f"origen{suf}.csv"), "w") as fh:
        fh.write("realizacion,carpeta,realizacion_original\n")
        for k, d, k0 in origen:
            fh.write(f"{k},{d},{k0}\n")
    st = A.pass2({"fps": fps}, real, suf, titulo + f" ({', '.join(s.partition(':')[0] for s in srcs)})")
    labels = [f"R{k0} de {d}" for _, d, k0 in origen]
    txt = A.tests([np.array([J["s"] for J in r["jumps"]]) for r in real], labels=labels)
    print(txt)
    with open(os.path.join(out, f"tests{suf}.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(f"{len(real)} realizaciones, {st['nsaltos']} saltos -> {out}")


if __name__ == "__main__":
    main()
