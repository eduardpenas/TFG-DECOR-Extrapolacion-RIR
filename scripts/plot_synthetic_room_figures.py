import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


WALL_ORDER = ["east", "west", "north", "south", "ceiling", "floor"]
WALL_COLORS = {
    "east": "#4e79a7",
    "west": "#f28e2b",
    "north": "#e15759",
    "south": "#76b7b2",
    "ceiling": "#59a14f",
    "floor": "#9c755f",
}
DEFAULT_CENTER_FREQS = np.array([125, 250, 500, 1000, 2000, 4000], dtype=np.float64)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f2f2f",
            "axes.labelcolor": "#2f2f2f",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "xtick.color": "#2f2f2f",
            "ytick.color": "#2f2f2f",
            "savefig.facecolor": "white",
        }
    )


def load_metadata(data_path: Path) -> list[dict[str, str]]:
    metadata_path = data_path / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontró metadata.csv en: {data_path}")

    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise RuntimeError("metadata.csv está vacío. Genera primero el dataset.")

    return rows


def pick_row(rows: list[dict[str, str]], sample_id: int) -> dict[str, str]:
    for row in rows:
        if int(row["sample_id"]) == sample_id:
            return row
    raise ValueError(f"No existe sample_id={sample_id} en metadata.csv")


def row_to_room(row: dict[str, str]) -> dict[str, np.ndarray | float | str]:
    room_dims = np.array(
        [float(row["room_length_m"]), float(row["room_width_m"]), float(row["room_height_m"])],
        dtype=np.float64,
    )
    source = np.array(
        [float(row["source_x_m"]), float(row["source_y_m"]), float(row["source_z_m"])],
        dtype=np.float64,
    )
    receiver = np.array(
        [float(row["receiver_x_m"]), float(row["receiver_y_m"]), float(row["receiver_z_m"])],
        dtype=np.float64,
    )
    room: dict[str, np.ndarray | float | str] = {
        "sample_id": int(row["sample_id"]),
        "room_dims": room_dims,
        "source": source,
        "receiver": receiver,
        "distance": float(row["source_receiver_distance_m"]),
        "mean_absorption": float(row["mean_absorption"]),
        "absorption_profile": row["absorption_profile"],
        "rt60": float(row["rt60_estimated_s"]),
    }

    center_freqs_raw = row.get("absorption_center_freqs_hz", "")
    if center_freqs_raw:
        room["center_freqs"] = np.asarray([float(value) for value in center_freqs_raw.split(",")], dtype=np.float64)
    else:
        room["center_freqs"] = DEFAULT_CENTER_FREQS

    materials: dict[str, np.ndarray] = {}
    for wall in WALL_ORDER:
        raw_coeffs = row.get(f"absorption_{wall}_coeffs", "")
        if raw_coeffs:
            materials[wall] = np.asarray([float(value) for value in raw_coeffs.split(",")], dtype=np.float64)
    room["materials"] = materials
    return room


def cube_edges(room_dims: np.ndarray) -> np.ndarray:
    lx, ly, lz = room_dims.tolist()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [lx, 0.0, 0.0],
            [lx, ly, 0.0],
            [0.0, ly, 0.0],
            [0.0, 0.0, lz],
            [lx, 0.0, lz],
            [lx, ly, lz],
            [0.0, ly, lz],
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return np.array([[vertices[i], vertices[j]] for i, j in edges], dtype=np.float64)


def plot_plan_xy(room: dict[str, np.ndarray | float | str], out_path: Path) -> None:
    room_dims = room["room_dims"]
    source = room["source"]
    receiver = room["receiver"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            float(room_dims[0]),
            float(room_dims[1]),
            fill=False,
            linewidth=2.2,
            edgecolor="#1f3b73",
        )
    )

    ax.scatter(source[0], source[1], s=120, c="#d62728", marker="*", label="Fuente", zorder=5)
    ax.scatter(receiver[0], receiver[1], s=90, c="#2ca02c", marker="o", label="Receptor", zorder=5)
    ax.plot([source[0], receiver[0]], [source[1], receiver[1]], color="#444444", linestyle="--", linewidth=1.4)

    ax.annotate("S", (source[0], source[1]), xytext=(6, 6), textcoords="offset points", fontsize=11, weight="bold")
    ax.annotate("R", (receiver[0], receiver[1]), xytext=(6, 6), textcoords="offset points", fontsize=11, weight="bold")

    ax.set_xlim(0.0, float(room_dims[0]))
    ax.set_ylim(0.0, float(room_dims[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Planta 2D de la sala | sample {room['sample_id']:05d} | perfil {room['absorption_profile']}"
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper right")

    info = (
        f"Lx={room_dims[0]:.2f} m | Ly={room_dims[1]:.2f} m | Lz={room_dims[2]:.2f} m | "
        f"d(S,R)={room['distance']:.2f} m | mean abs={room['mean_absorption']:.3f}"
    )
    fig.text(0.5, 0.01, info, ha="center", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_room_3d(room: dict[str, np.ndarray | float | str], out_path: Path) -> None:
    room_dims = room["room_dims"]
    source = room["source"]
    receiver = room["receiver"]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    segments = cube_edges(room_dims)
    ax.add_collection3d(Line3DCollection(segments, colors="#1f3b73", linewidths=1.8, alpha=0.95))
    ax.scatter(*source, s=80, c="#d62728", marker="*", depthshade=True, label="Fuente")
    ax.scatter(*receiver, s=65, c="#2ca02c", marker="o", depthshade=True, label="Receptor")
    ax.plot(
        [source[0], receiver[0]],
        [source[1], receiver[1]],
        [source[2], receiver[2]],
        color="#444444",
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlim(0.0, float(room_dims[0]))
    ax.set_ylim(0.0, float(room_dims[1]))
    ax.set_zlim(0.0, float(room_dims[2]))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Vista 3D de la sala | sample {room['sample_id']:05d}")
    ax.view_init(elev=22, azim=-55)
    ax.legend(frameon=False, loc="upper left")

    info = (
        f"Lx={room_dims[0]:.2f} m | Ly={room_dims[1]:.2f} m | Lz={room_dims[2]:.2f} m | "
        f"perfil={room['absorption_profile']}"
    )
    fig.text(0.5, 0.01, info, ha="center", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_statistics(rows: list[dict[str, str]], out_path: Path) -> None:
    lengths = np.array([float(row["room_length_m"]) for row in rows], dtype=np.float64)
    widths = np.array([float(row["room_width_m"]) for row in rows], dtype=np.float64)
    heights = np.array([float(row["room_height_m"]) for row in rows], dtype=np.float64)
    distances = np.array([float(row["source_receiver_distance_m"]) for row in rows], dtype=np.float64)
    mean_abs = np.array([float(row["mean_absorption"]) for row in rows], dtype=np.float64)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Estadísticas geométricas del dataset sintético", fontsize=14, fontweight="bold")

    bins = 28
    color = "#3f51b5"

    axs[0, 0].hist(lengths, bins=bins, color=color, alpha=0.85, edgecolor="white")
    axs[0, 0].set_title("Longitud de sala (X)")
    axs[0, 0].set_xlabel("m")
    axs[0, 0].set_ylabel("Frecuencia")

    axs[0, 1].hist(widths, bins=bins, color="#009688", alpha=0.85, edgecolor="white")
    axs[0, 1].set_title("Anchura de sala (Y)")
    axs[0, 1].set_xlabel("m")
    axs[0, 1].set_ylabel("Frecuencia")

    axs[1, 0].hist(heights, bins=bins, color="#ff9800", alpha=0.85, edgecolor="white")
    axs[1, 0].set_title("Altura de sala (Z)")
    axs[1, 0].set_xlabel("m")
    axs[1, 0].set_ylabel("Frecuencia")

    axs[1, 1].hist(distances, bins=bins, color="#8e24aa", alpha=0.85, edgecolor="white", label="Distancia S-R")
    axs[1, 1].hist(mean_abs, bins=bins, color="#d32f2f", alpha=0.5, edgecolor="white", label="Absorción media")
    axs[1, 1].set_title("Distancia fuente-receptor y absorción media")
    axs[1, 1].set_xlabel("Valor")
    axs[1, 1].set_ylabel("Frecuencia")
    axs[1, 1].legend(frameon=False)

    for ax in axs.flat:
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_material_curves(room: dict[str, np.ndarray | float | str], out_path: Path) -> None:
    materials = room.get("materials", {})
    center_freqs = room.get("center_freqs", DEFAULT_CENTER_FREQS)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    if not materials:
        ax.text(
            0.5,
            0.5,
            "Detalle de materiales no disponible\nen este metadata.csv",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        for wall in WALL_ORDER:
            coeffs = materials.get(wall)
            if coeffs is None:
                continue
            ax.plot(
                center_freqs,
                coeffs,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=WALL_COLORS[wall],
                label=wall.capitalize(),
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(center_freqs)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        ax.set_xlabel("Frecuencia central (Hz)")
        ax.set_ylabel("Coeficiente de absorción")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Coeficientes de absorción por pared")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    info = (
        f"Perfil={room['absorption_profile']} | mean abs={room['mean_absorption']:.3f} | "
        f"RT60={room['rt60']:.2f} s"
    )
    fig.text(0.5, 0.02, info, ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_absorption_profile_summary(rows: list[dict[str, str]], out_path: Path) -> None:
    profiles = [row["absorption_profile"] for row in rows]
    values, counts = np.unique(profiles, return_counts=True)
    order = np.argsort(-counts)
    values = values[order]
    counts = counts[order]

    mean_abs = np.array([float(row["mean_absorption"]) for row in rows], dtype=np.float64)
    profile_to_mean: dict[str, np.ndarray] = {}
    for profile in values:
        profile_to_mean[profile] = np.array(
            [float(row["mean_absorption"]) for row in rows if row["absorption_profile"] == profile],
            dtype=np.float64,
        )

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Variabilidad de los materiales sintéticos", fontsize=14, fontweight="bold")

    axs[0].bar(values, counts, color="#455a64")
    axs[0].set_title("Distribución de perfiles de absorción")
    axs[0].set_xlabel("Perfil")
    axs[0].set_ylabel("Número de muestras")
    axs[0].tick_params(axis="x", rotation=20)

    box_data = [profile_to_mean[profile] for profile in values]
    axs[1].boxplot(box_data, labels=values, showmeans=True)
    axs[1].set_title("Absorción media por perfil")
    axs[1].set_xlabel("Perfil")
    axs[1].set_ylabel("mean_absorption")
    axs[1].grid(alpha=0.2, axis="y")

    fig.text(0.5, 0.01, f"Absorción media global: {float(np.mean(mean_abs)):.3f}", ha="center", fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_memory_ready_figure(room: dict[str, np.ndarray | float | str], rows: list[dict[str, str]], out_path: Path) -> None:
    lengths = np.array([float(row["room_length_m"]) for row in rows], dtype=np.float64)
    widths = np.array([float(row["room_width_m"]) for row in rows], dtype=np.float64)
    heights = np.array([float(row["room_height_m"]) for row in rows], dtype=np.float64)
    distances = np.array([float(row["source_receiver_distance_m"]) for row in rows], dtype=np.float64)
    materials = room.get("materials", {})
    center_freqs = room.get("center_freqs", DEFAULT_CENTER_FREQS)

    fig = plt.figure(figsize=(17, 15))
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.0, 0.9], hspace=0.38, wspace=0.22)

    ax_plan = fig.add_subplot(outer[0, 0])
    ax_3d = fig.add_subplot(outer[0, 1], projection="3d")
    ax_mat = fig.add_subplot(outer[1, :])
    stats_grid = outer[2, :].subgridspec(1, 4, wspace=0.28)
    ax_len = fig.add_subplot(stats_grid[0, 0])
    ax_wid = fig.add_subplot(stats_grid[0, 1])
    ax_hei = fig.add_subplot(stats_grid[0, 2])
    ax_dis = fig.add_subplot(stats_grid[0, 3])

    room_dims = room["room_dims"]
    source = room["source"]
    receiver = room["receiver"]

    ax_plan.add_patch(
        plt.Rectangle((0.0, 0.0), float(room_dims[0]), float(room_dims[1]), fill=False, linewidth=2.4, edgecolor="#1f3b73")
    )
    ax_plan.scatter(source[0], source[1], s=150, c="#c7254e", marker="*", label="Fuente", zorder=5)
    ax_plan.scatter(receiver[0], receiver[1], s=110, c="#2f855a", marker="o", label="Receptor", zorder=5)
    ax_plan.plot([source[0], receiver[0]], [source[1], receiver[1]], color="#444444", linestyle="--", linewidth=1.5)
    ax_plan.set_xlim(0.0, float(room_dims[0]))
    ax_plan.set_ylim(0.0, float(room_dims[1]))
    ax_plan.set_aspect("equal", adjustable="box")
    ax_plan.set_xlabel("X (m)")
    ax_plan.set_ylabel("Y (m)")
    ax_plan.set_title("Planta 2D")
    ax_plan.legend(loc="upper right")
    ax_plan.text(
        0.03,
        0.03,
        f"Lx={room_dims[0]:.2f} m\nLy={room_dims[1]:.2f} m\nLz={room_dims[2]:.2f} m",
        transform=ax_plan.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cfd8dc", alpha=0.95),
    )

    segments = cube_edges(room_dims)
    ax_3d.add_collection3d(Line3DCollection(segments, colors="#1f3b73", linewidths=1.8, alpha=0.95))
    ax_3d.scatter(*source, s=90, c="#c7254e", marker="*", depthshade=True, label="Fuente")
    ax_3d.scatter(*receiver, s=70, c="#2f855a", marker="o", depthshade=True, label="Receptor")
    ax_3d.plot(
        [source[0], receiver[0]],
        [source[1], receiver[1]],
        [source[2], receiver[2]],
        color="#444444",
        linestyle="--",
        linewidth=1.2,
    )
    ax_3d.set_xlim(0.0, float(room_dims[0]))
    ax_3d.set_ylim(0.0, float(room_dims[1]))
    ax_3d.set_zlim(0.0, float(room_dims[2]))
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("Vista 3D")
    ax_3d.view_init(elev=22, azim=-55)
    ax_3d.legend(loc="upper left")

    if materials:
        for wall in WALL_ORDER:
            coeffs = materials.get(wall)
            if coeffs is None:
                continue
            ax_mat.plot(
                center_freqs,
                coeffs,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=WALL_COLORS[wall],
                label=wall.capitalize(),
            )
        ax_mat.set_xscale("log", base=2)
        ax_mat.set_xticks(center_freqs)
        ax_mat.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        ax_mat.set_ylim(0.0, 1.05)
        ax_mat.set_ylabel("Absorción")
        ax_mat.set_xlabel("Frecuencia central (Hz)")
        ax_mat.set_title("Coeficientes de absorción por pared")
        ax_mat.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    else:
        ax_mat.text(
            0.5,
            0.5,
            "Detalle de materiales no disponible\nen este metadata.csv",
            ha="center",
            va="center",
            transform=ax_mat.transAxes,
            fontsize=12,
        )
        ax_mat.set_axis_off()

    for ax, values, title, color in [
        (ax_len, lengths, "Longitud (X)", "#3f51b5"),
        (ax_wid, widths, "Anchura (Y)", "#009688"),
        (ax_hei, heights, "Altura (Z)", "#ff9800"),
        (ax_dis, distances, "Distancia S-R", "#8e24aa"),
    ]:
        ax.hist(values, bins=24, color=color, alpha=0.86, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Frecuencia")
        ax.grid(alpha=0.22)

    ax_len.set_xlabel("m")
    ax_wid.set_xlabel("m")
    ax_hei.set_xlabel("m")
    ax_dis.set_xlabel("m")

    fig.suptitle(
        f"Dataset sintético | sample {room['sample_id']:05d} | perfil {room['absorption_profile']} | RT60 {room['rt60']:.2f} s",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        f"Fuente-receptor: {room['distance']:.2f} m | Absorción media: {room['mean_absorption']:.3f}",
        ha="center",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera figuras de sala para el dataset sintético: planta, 3D y estadísticas."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/raw"), help="Ruta al dataset sintético")
    parser.add_argument("--sample-id", type=int, default=0, help="ID de muestra para las figuras geométricas")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Figures/synthetic_rooms"),
        help="Carpeta de salida para los PNG generados",
    )
    parser.add_argument(
        "--stats-samples",
        type=int,
        default=500,
        help="Número de filas a usar para las estadísticas (0 = todas)",
    )
    return parser.parse_args()


def main() -> None:
    configure_style()
    args = parse_args()
    rows = load_metadata(args.data_path)
    row = row_to_room(pick_row(rows, args.sample_id))

    stats_rows = rows if args.stats_samples <= 0 else rows[: min(args.stats_samples, len(rows))]

    plan_path = args.out_dir / f"room_plan_xy_sample_{row['sample_id']:05d}.png"
    room3d_path = args.out_dir / f"room_3d_sample_{row['sample_id']:05d}.png"
    materials_path = args.out_dir / f"room_materials_sample_{row['sample_id']:05d}.png"
    stats_path = args.out_dir / "room_geometry_statistics.png"
    absorption_path = args.out_dir / "room_absorption_statistics.png"
    memory_path = args.out_dir / f"room_memory_figure_sample_{row['sample_id']:05d}.png"

    plot_plan_xy(row, plan_path)
    plot_room_3d(row, room3d_path)
    plot_material_curves(row, materials_path)
    plot_statistics(stats_rows, stats_path)
    plot_absorption_profile_summary(rows, absorption_path)
    plot_memory_ready_figure(row, stats_rows, memory_path)

    print(f"Figura guardada en: {plan_path}")
    print(f"Figura guardada en: {room3d_path}")
    print(f"Figura guardada en: {materials_path}")
    print(f"Figura guardada en: {stats_path}")
    print(f"Figura guardada en: {absorption_path}")
    print(f"Figura guardada en: {memory_path}")


if __name__ == "__main__":
    main()