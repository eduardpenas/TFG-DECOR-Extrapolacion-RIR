import argparse
import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

def iter_files_sorted(root: Path) -> Iterable[Path]:
    files = []
    for current_root, _, filenames in os.walk(root, followlinks=True):
        base = Path(current_root)
        for filename in filenames:
            files.append(base / filename)
    files.sort(key=lambda p: str(p.relative_to(root)).replace("\\", "/"))
    return files


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audit_data(data_root: Path, bird_subdir: Path) -> Counter:
    print(f"[AUDIT] Data root: {data_root}")
    print(f"[AUDIT] Bird root: {bird_subdir}")

    if bird_subdir.exists():
        print("[AUDIT] Directorios relevantes de BIRD (profundidad <= 3):")
        directories = set()
        for current_root, dirnames, _ in os.walk(bird_subdir, followlinks=True):
            base = Path(current_root)
            directories.add(base)
            for dirname in dirnames:
                directories.add(base / dirname)

        for directory in sorted(directories):
            rel = directory.relative_to(data_root)
            if len(rel.parts) <= 3:
                print(f" - {rel.as_posix()}")
    else:
        print("[AUDIT] Aviso: no existe la ruta BIRD indicada.")

    ext_counter: Counter = Counter()
    for file_path in iter_files_sorted(data_root):
        suffix = file_path.suffix.lower().lstrip(".") or "<sin_extension>"
        ext_counter[suffix] += 1

    print("[AUDIT] Conteo de archivos por extensión:")
    for ext, count in ext_counter.most_common():
        print(f" - .{ext}: {count}")

    return ext_counter


def write_manifest(data_root: Path, manifest_path: Path) -> int:
    files = list(iter_files_sorted(data_root))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as output:
        for file_path in files:
            hash_value = sha256_file(file_path)
            if file_path.is_absolute():
                rel = file_path.relative_to(Path.cwd())
            else:
                rel = file_path
            output.write(f"{hash_value}  {rel.as_posix()}\n")

    print(f"[HASH] Manifest escrito en: {manifest_path}")
    print(f"[HASH] Total archivos hasheados: {len(files)}")
    return len(files)


def verify_manifest(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe el manifest: {manifest_path}")

    ok = True
    total = 0

    with manifest_path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line:
                continue

            expected_hash, rel_path = line.split("  ", maxsplit=1)
            file_path = Path.cwd() / rel_path
            total += 1

            if not file_path.exists():
                print(f"[VERIFY][MISSING] {rel_path}")
                ok = False
                continue

            current_hash = sha256_file(file_path)
            if current_hash != expected_hash:
                print(f"[VERIFY][MISMATCH] {rel_path}")
                ok = False

    print(f"[VERIFY] Archivos verificados: {total}")
    print(f"[VERIFY] Estado: {'OK' if ok else 'ERROR'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita data/ y genera/valida manifest SHA256")
    parser.add_argument("--data-root", type=str, default="data", help="Ruta raíz de datos.")
    parser.add_argument("--bird-root", type=str, default="data/BIRD", help="Ruta de BIRD dentro de data/.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data_hash_manifest.sha256",
        help="Ruta de salida del manifest SHA256.",
    )
    parser.add_argument("--verify", action="store_true", help="Verifica un manifest existente.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    bird_root = Path(args.bird_root)
    manifest_path = Path(args.manifest)

    if args.verify:
        verify_manifest(manifest_path)
        return

    if not data_root.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {data_root}")

    audit_data(data_root=data_root, bird_subdir=bird_root)
    write_manifest(data_root=data_root, manifest_path=manifest_path)


if __name__ == "__main__":
    main()
