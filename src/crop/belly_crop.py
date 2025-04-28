import cv2, numpy as np, pandas as pd, scipy.interpolate as si
from pathlib import Path
from tqdm import tqdm

# --- константы ---
MID_IDX   = np.array([2, 5, 8, 11, 14, 17, 20])         # 1-based
TARGET_W, TARGET_H = 400, 800                           # финальное разрешение

# --- вспомогательные функции -------------------------------------------------
def load_points(csv_file):
    """Вернём три (7×2) массива: mid, left, right."""
    df = pd.read_csv(csv_file, header=[0, 1, 2])        # scorer/bodypart/coord
    scorer = df.columns.get_level_values(0)[0]
    pts = {}
    for i in range(1, 22):                              # point1…point21
        x = df[(scorer, f"point{i}", "x")].iloc[0]
        y = df[(scorer, f"point{i}", "y")].iloc[0]
        pts[i] = np.array([x, y], float)

    mid   = np.vstack([pts[i]          for i in MID_IDX])
    left  = np.vstack([pts[i-1]        for i in MID_IDX])
    right = np.vstack([pts[i+1]        for i in MID_IDX])
    return mid, left, right


def build_splines(mid, left, right):
    """Сплайны: центр C(s), нормаль n(s), половина ширины w(s)."""
    tck_mid, _ = si.splprep(mid.T, s=0)
    tck_L,   _ = si.splprep(left.T, s=0)
    tck_R,   _ = si.splprep(right.T, s=0)

    def C(s): return np.stack(si.splev(s, tck_mid), axis=-1)
    def T(s): return np.stack(si.splev(s, tck_mid, der=1), axis=-1)
    def n(s):
        t = T(s)
        n = np.stack([-t[..., 1], t[..., 0]], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return n
    def w(s):
        L = np.stack(si.splev(s, tck_L), axis=-1)
        R = np.stack(si.splev(s, tck_R), axis=-1)
        return 0.5 * np.linalg.norm(R - L, axis=-1)

    return C, n, w


def unwrap_belly(frame, C, nfunc, wfunc, H=TARGET_H, W=TARGET_W):
    """Возвращает прямоугольник WxH «живота»."""
    u, v = np.mgrid[0:H, 0:W].astype(np.float32)
    u /= (H - 1)                       # 0…1 вдоль тела
    v = v / (W - 1) * 2 - 1            # -1…1 поперёк

    c  = C(u.ravel())
    n  = nfunc(u.ravel())
    xy = c + n * (wfunc(u.ravel())[:, None] * v.ravel()[:, None])

    map_x = xy[:, 0].reshape(H, W).astype(np.float32)
    map_y = xy[:, 1].reshape(H, W).astype(np.float32)

    return cv2.remap(
        frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

# --- основной проход ---------------------------------------------------------
def process_folder(img_dir: Path, csv_dir: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    for csv_file in tqdm(sorted(csv_dir.glob("*.csv")), desc="Cropping"):
        name = csv_file.stem
        img_path = img_dir / f"{name}.png"              # тот же базовый файл
        if not img_path.exists():
            print(f"⚠  {img_path} not found, skip.")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"⚠  Cannot read {img_path}")
            continue

        mid, left, right = load_points(csv_file)
        C, nfunc, wfunc = build_splines(mid, left, right)
        rect = unwrap_belly(frame, C, nfunc, wfunc)
        cv2.imwrite(str(out_dir / f"{name}.png"), rect)

    print(f"✔ Done. Results in {out_dir}")

# --- точка входа -------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Dynamic belly crop from DLC keypoints"
    )
    parser.add_argument("images", help="Папка с исходными кадрами (.png)")
    parser.add_argument("csvs",   help="Папка с CSV от DeepLabCut")
    parser.add_argument("--out",  default="results", help="Куда сохранить кропы")
    args = parser.parse_args()

    process_folder(Path(args.images), Path(args.csvs), Path(args.out))