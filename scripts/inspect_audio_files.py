"""
Inspecciona los archivos de audio del dataset (buscando .flaac, .flac, .wav)
y muestra: ruta, extensión, frecuencia de muestreo, canales (mono/stereo) y duración.

Uso:
    python scripts/inspect_audio_files.py [--root data/BIRD] [--ext .flaac .flac .wav]
    python scripts/inspect_audio_files.py --root data/BIRD --ext .flaac
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sep(char="─", w=72):
    print(char * w)


def inspect_file(path: Path) -> dict | None:
    """Lee metadatos de un archivo de audio usando soundfile (sin FFmpeg)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return {
            "path": path,
            "ext": path.suffix.lower(),
            "sr": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_ms": info.frames / info.samplerate * 1000,
            "error": None,
        }
    except Exception as e:
        return {
            "path": path,
            "ext": path.suffix.lower(),
            "sr": None,
            "channels": None,
            "frames": None,
            "duration_ms": None,
            "error": str(e),
        }


def collect_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    # Resolver symlinks para que os.walk pueda atravesarlos
    import os
    real_root = root.resolve()
    files = []
    for dirpath, _, filenames in os.walk(real_root, followlinks=True):
        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                files.append(Path(dirpath) / fname)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Inspecciona archivos de audio del dataset")
    parser.add_argument("--root", type=str, default="data/BIRD",
                        help="Directorio raíz de búsqueda.")
    parser.add_argument("--ext", nargs="+", default=[".flaac", ".flac", ".wav"],
                        help="Extensiones a buscar (por defecto: .flaac .flac .wav).")
    parser.add_argument("--max-show", type=int, default=50,
                        help="Máximo de ficheros a mostrar individualmente (0 = todos).")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: El directorio '{root}' no existe.")
        sys.exit(1)

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext)

    print(f"Buscando {exts} en: {root.resolve()}")
    files = collect_files(root, exts)

    _sep("═")
    print(f"  Archivos encontrados: {len(files)}")
    _sep("═")

    if not files:
        print("  No se encontraron archivos con las extensiones indicadas.")
        return

    # Inspeccionar
    records = [inspect_file(f) for f in files]
    errors = [r for r in records if r["error"]]
    ok = [r for r in records if not r["error"]]

    # ── Tabla detallada ───────────────────────────────────────────────────────
    limit = len(records) if args.max_show == 0 else args.max_show
    print(f"\n{'ARCHIVO':<45}  {'EXT':<7}  {'SR':>7}  {'CH':>4}  {'DURACIÓN':>10}  {'CANALES'}")
    _sep()
    for r in records[:limit]:
        if r["error"]:
            print(f"  {str(r['path'].name):<43}  {r['ext']:<7}  {'ERROR':>7}  {''!s:>4}  {''!s:>10}  {r['error'][:35]}")
        else:
            ch_str = "mono" if r["channels"] == 1 else f"stereo({r['channels']}ch)" if r["channels"] == 2 else f"{r['channels']}ch"
            dur_str = f"{r['duration_ms']:.0f} ms"
            print(f"  {str(r['path'].name):<43}  {r['ext']:<7}  {r['sr']:>7}  {r['channels']:>4}  {dur_str:>10}  {ch_str}")

    if limit < len(records):
        print(f"  ... y {len(records) - limit} archivos más (usa --max-show 0 para ver todos)")

    # ── Resumen estadístico ───────────────────────────────────────────────────
    _sep("═")
    print("RESUMEN ESTADÍSTICO")
    _sep("═")

    # Por extensión
    ext_counts = Counter(r["ext"] for r in records)
    print("\n  Por extensión:")
    for ext, count in sorted(ext_counts.items()):
        print(f"    {ext:<8}: {count:>6} archivos")

    # Sample rates (solo los OK)
    if ok:
        sr_counts = Counter(r["sr"] for r in ok)
        print("\n  Sample rates:")
        for sr, count in sorted(sr_counts.items()):
            pct = 100.0 * count / len(ok)
            compat = "✓ (nativo 48 kHz)" if sr == 48000 else f"→ se resamulea a 48 kHz"
            print(f"    {sr:>7} Hz: {count:>6} archivos  ({pct:.1f}%)  {compat}")

        # Canales
        ch_counts = Counter(r["channels"] for r in ok)
        print("\n  Canales:")
        for ch, count in sorted(ch_counts.items()):
            pct = 100.0 * count / len(ok)
            ch_str = "mono" if ch == 1 else "stereo" if ch == 2 else f"{ch} canales"
            print(f"    {ch_str:<10}: {count:>6} archivos  ({pct:.1f}%)")

        # Duraciones
        durations = [r["duration_ms"] for r in ok]
        print(f"\n  Duración:")
        print(f"    mín : {min(durations):.0f} ms")
        print(f"    máx : {max(durations):.0f} ms")
        print(f"    media: {sum(durations)/len(durations):.0f} ms")

    # Errores
    if errors:
        _sep()
        print(f"\n  ERRORES ({len(errors)} archivos no legibles):")
        for r in errors:
            print(f"    {r['path'].name}: {r['error']}")

    _sep("═")

    # Advertencias relevantes para la pipeline
    if ok:
        non_48k = [r for r in ok if r["sr"] != 48000]
        stereo  = [r for r in ok if r["channels"] and r["channels"] > 1]
        short   = [r for r in ok if r["duration_ms"] and r["duration_ms"] < 1000]

        if non_48k or stereo or short:
            print("AVISOS PIPELINE:")
            if non_48k:
                srs = sorted(set(r["sr"] for r in non_48k))
                print(f"  ⚠  {len(non_48k)} archivos no están a 48 kHz {srs} → BirdDataset los resamulea automáticamente")
            if stereo:
                print(f"  ⚠  {len(stereo)} archivos son stereo → BirdDataset los convierte a mono automáticamente")
            if short:
                print(f"  ⚠  {len(short)} archivos duran <1000 ms → se aplicará padding en BirdDataset")
            _sep("═")


if __name__ == "__main__":
    main()
