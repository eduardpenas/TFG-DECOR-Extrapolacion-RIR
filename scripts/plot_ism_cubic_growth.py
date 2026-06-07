import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Orden de reflexion maximo (1..15)
    n = np.arange(1, 16)

    # Numero total de fuentes virtuales en ISM 3D.
    # Crecimiento cubico dominante O(N^3).
    num_fuentes = (4.0 / 3.0) * n**3 + 2.0 * n**2 + (8.0 / 3.0) * n + 1.0

    plt.figure(figsize=(8, 5))
    plt.plot(
        n,
        num_fuentes,
        marker="o",
        linestyle="-",
        color="#1f77b4",
        linewidth=2,
        markersize=6,
    )

    plt.title("Crecimiento de fuentes virtuales en el metodo ISM (3D)", fontsize=14)
    plt.xlabel("Orden de Reflexion (N)", fontsize=12)
    plt.ylabel("Numero Total de Fuentes Virtuales", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_path = "Figures/ism_cubic_growth.png"
    plt.savefig(out_path, dpi=300)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
