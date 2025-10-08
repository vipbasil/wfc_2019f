import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import argparse, os, re, random, json, io
from PIL import Image, ImageDraw, ImageFont

ROT_COUNTS = {"X":1, "I":2, "L":4, "T":4, "\\":2, "/":2}

def clean_xml_text(txt: str) -> str:
    # FIX: real newline, strip conflict markers + BOM
    lines = []
    for line in txt.splitlines():
        if line.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            continue
        lines.append(line)
    return "\n".join(lines).lstrip("\ufeff")

@dataclass(frozen=True)
class Variant:
    name: str
    rot: int
    def key(self) -> str:
        return f"{self.name}@{self.rot}"

@dataclass
class TileDef:
    name: str
    symmetry: str
    rot_count: int
    weight: float

class RuleSet:
    def __init__(self, variants: List[str]):
        self.variants = set(variants)
        self.right_of: Dict[str, Set[str]] = {v: set() for v in variants}
        self.below_of: Dict[str, Set[str]] = {v: set() for v in variants}
    def add_right(self, left: str, right: str):
        if left in self.variants and right in self.variants:
            self.right_of[left].add(right)
    def add_below(self, top: str, bottom: str):
        if top in self.variants and bottom in self.variants:
            self.below_of[top].add(bottom)

def parse_tile_token(tok: str):
    # robust: normalize NBSP/tabs/spaces, allow optional trailing integer
    tok = tok.strip().replace("\u00A0", " ")
    if not tok:
        raise ValueError("Empty tile token")
    parts = tok.rsplit(None, 1)
    if len(parts) == 2 and parts[1].isdigit():
        name, rot = parts[0], int(parts[1])
    else:
        name, rot = tok, None
    return name, rot

def load_tileset(xml_path: str):
    # tolerate BOM / invalid bytes
    with io.open(xml_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    raw = clean_xml_text(raw)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        # Helpful diagnostics
        raise RuntimeError(f"XML parse error at line {e.position[0]} col {e.position[1]}: {e}") from e

    if root.tag != "set":
        raise RuntimeError("Root element must be <set>")

    tiles_node = root.find("tiles")
    neigh_node = root.find("neighbors")
    if tiles_node is None or neigh_node is None:
        raise RuntimeError("XML must contain <tiles> and <neighbors>")

    tile_defs: Dict[str, TileDef] = {}
    for t in tiles_node.findall("tile"):
        name = t.attrib["name"]
        symmetry = t.attrib.get("symmetry", "X")
        if symmetry not in ROT_COUNTS:
            raise RuntimeError(f"Unknown symmetry '{symmetry}' for tile '{name}'")
        rot_count = ROT_COUNTS[symmetry]
        weight = float(t.attrib.get("weight", "1.0"))
        tile_defs[name] = TileDef(name, symmetry, rot_count, weight)

    variants: List[str] = []
    for name, td in tile_defs.items():
        for r in range(td.rot_count):
            variants.append(f"{name}@{r}")

    explicit_lr: List[Tuple[str,str]] = []
    explicit_tb: List[Tuple[str,str]] = []

    for n in neigh_node.findall("neighbor"):
        if "left" in n.attrib and "right" in n.attrib:
            Lname, Lrot = parse_tile_token(n.attrib["left"])
            Rname, Rrot = parse_tile_token(n.attrib["right"])
            if Lname not in tile_defs or Rname not in tile_defs:
                raise RuntimeError(f"Neighbor references unknown tile: '{Lname}' or '{Rname}'")
            Lrot = 0 if Lrot is None else (Lrot % tile_defs[Lname].rot_count)
            Rrot = 0 if Rrot is None else (Rrot % tile_defs[Rname].rot_count)
            explicit_lr.append((f"{Lname}@{Lrot}", f"{Rname}@{Rrot}"))
        if "top" in n.attrib and "bottom" in n.attrib:
            Tname, Trot = parse_tile_token(n.attrib["top"])
            Bname, Brot = parse_tile_token(n.attrib["bottom"])
            if Tname not in tile_defs or Bname not in tile_defs:
                raise RuntimeError(f"Neighbor references unknown tile: '{Tname}' or '{Bname}'")
            Trot = 0 if Trot is None else (Trot % tile_defs[Tname].rot_count)
            Brot = 0 if Brot is None else (Brot % tile_defs[Bname].rot_count)
            explicit_tb.append((f"{Tname}@{Trot}", f"{Bname}@{Brot}"))
    return tile_defs, variants, explicit_lr, explicit_tb

def compile_rules(tile_defs, variants, explicit_lr, explicit_tb,
                  auto_rotate=True, auto_reciprocal=True):
    rs = RuleSet(variants)

    def add_lr(a, b):
        rs.add_right(a, b)
        if auto_reciprocal:
            rs.add_right(b, a)

    def add_tb(a, b):
        rs.add_below(a, b)
        if auto_reciprocal:
            rs.add_below(b, a)

    # direct pairs from XML
    for a, b in explicit_lr: add_lr(a, b)
    for a, b in explicit_tb: add_tb(a, b)

    # infer rotated pairs (turn L/R into T/B on odd 90° rotations)
    if auto_rotate:
        def rotate_key(key, k):
            name, rot_s = key.split("@"); rot = int(rot_s)
            rc = tile_defs[name].rot_count
            if rc == 1: new_rot = 0
            elif rc == 2: new_rot = (rot + k) % 2
            else: new_rot = (rot + k) % 4
            return f"{name}@{new_rot}"

        for a, b in list(explicit_lr):
            for k in (0, 1, 2, 3):
                ak, bk = rotate_key(a, k), rotate_key(b, k)
                if k != 0 and ak == a and bk == b:
                    continue
                (add_lr if k % 2 == 0 else add_tb)(ak, bk)

        for a, b in list(explicit_tb):
            for k in (0, 1, 2, 3):
                ak, bk = rotate_key(a, k), rotate_key(b, k)
                if k != 0 and ak == a and bk == b:
                    continue
                (add_tb if k % 2 == 0 else add_lr)(ak, bk)

    return rs

class CSPGrid:
    def __init__(self, W, H, variants, tile_defs, rules, seed=None,
                 choice="weighted", temperature=1.0):
        self.W, self.H = W, H
        self.variants = variants
        self.tile_defs = tile_defs
        self.rules = rules
        self.rng = random.Random(seed)
        self.choice = choice          # "weighted" or "argmax"
        self.temperature = temperature
        # state
        self.grid   = [[None for _ in range(W)] for _ in range(H)]
        self.domain = [[set(variants) for _ in range(W)] for _ in range(H)]
        self.weights = {k: tile_defs[k.split("@")[0]].weight for k in variants}

    def neighbors(self, r, c):
        if c+1 < self.W: yield (r, c+1, 'R')
        if c-1 >= 0:     yield (r, c-1, 'L')
        if r+1 < self.H: yield (r+1, c, 'D')
        if r-1 >= 0:     yield (r-1, c, 'U')

    def propagate(self):
        changed = True
        while changed:
            changed = False
            for r in range(self.H):
                for c in range(self.W):
                    if self.grid[r][c] is not None:
                        self.domain[r][c] = {self.grid[r][c]}
                        continue
                    dom = self.domain[r][c]
                    new_dom = set()
                    for v in dom:
                        ok = True
                        for rr, cc, d in self.neighbors(r, c):
                            nd = self.domain[rr][cc]
                            if self.grid[rr][cc] is not None:
                                nd = {self.grid[rr][cc]}
                            if d == 'R':
                                if not (nd & self.rules.right_of[v]): ok = False; break
                            elif d == 'L':
                                if not any(v in self.rules.right_of[u] for u in nd): ok = False; break
                            elif d == 'D':
                                if not (nd & self.rules.below_of[v]): ok = False; break
                            elif d == 'U':
                                if not any(v in self.rules.below_of[u] for u in nd): ok = False; break
                        if ok: new_dom.add(v)
                    if new_dom != dom:
                        self.domain[r][c] = new_dom
                        changed = True
                        if not new_dom:
                            return False
        return True

    def select_cell(self):
        # MRV + random tie-break; return None when grid full
        best = []
        best_sz = 10**9
        for r in range(self.H):
            for c in range(self.W):
                if self.grid[r][c] is None:
                    sz = len(self.domain[r][c])
                    if sz < best_sz:
                        best, best_sz = [(r, c)], sz
                    elif sz == best_sz:
                        best.append((r, c))
        return self.rng.choice(best) if best else None

    def _snapshot(self):
        # deep copy of the set grid + value grid
        return [[set(s) for s in row] for row in self.domain], [row[:] for row in self.grid]

    def _restore(self, dom_snap, grid_snap):
        self.domain = [[set(s) for s in row] for row in dom_snap]
        self.grid   = [row[:] for row in grid_snap]

    def _value_order(self, opts):
        if self.choice == "argmax":
            # deterministic but with a tiny shuffle to break ties
            opts = list(opts)
            self.rng.shuffle(opts)
            opts.sort(key=lambda k: -self.weights[k])
            return opts
        # weighted sampling without replacement (temperature controls greediness)
        T = max(1e-9, float(self.temperature))
        remaining = set(opts)
        order = []
        while remaining:
            pool = list(remaining)
            w = [max(1e-9, self.weights[o]) ** (1.0 / T) for o in pool]
            pick = self.rng.choices(pool, weights=w, k=1)[0]
            order.append(pick)
            remaining.remove(pick)
        return order

    def solve(self):
        if not self.propagate():
            return False
        def dfs():
            cell = self.select_cell()
            if cell is None:
                return True
            r, c = cell
            order = self._value_order(self.domain[r][c])
            for v in order:
                dom_snap, grid_snap = self._snapshot()
                self.grid[r][c] = v
                if self.propagate() and dfs():
                    return True
                self._restore(dom_snap, grid_snap)
                self.grid[r][c] = None
            return False
        return dfs()


def load_tile_images(tile_defs, tile_dir):
    imgs = {}
    if not tile_dir: return imgs
    for name in tile_defs.keys():
        for cand in (f"{name}.png", f"{name}.PNG", f"{name}.jpg", f"{name}.jpeg"):
            path = os.path.join(tile_dir, cand)
            if os.path.isfile(path):
                imgs[name] = Image.open(path).convert("RGBA")
                break
    return imgs

def rotate_image(img, rot):
    return img if rot == 0 else img.rotate(-90*rot, expand=False)

def render_grid(grid, tile_defs, tile_imgs, out_path):
    H, W = len(grid), len(grid[0])
    TW, TH = (next(iter(tile_imgs.values())).size if tile_imgs else (32, 32))
    canvas = Image.new("RGBA", (W*TW, H*TH), (0,0,0,0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 10)
    except:
        font = ImageFont.load_default()
    for r in range(H):
        for c in range(W):
            key = grid[r][c]
            if key is None: continue
            name, rot = key.split("@")[0], int(key.split("@")[1])
            if name in tile_imgs:
                img = rotate_image(tile_imgs[name], rot)
                canvas.paste(img, (c*TW, r*TH), img if img.mode=="RGBA" else None)
            else:
                color_seed = sum(ord(ch) for ch in name) % 255
                color = (color_seed, 128, 255-color_seed, 255)
                x0, y0 = c*TW, r*TH
                draw.rectangle([x0, y0, x0+TW-1, y0+TH-1], fill=color, outline=(40,40,40,255))
                draw.multiline_text((x0+3, y0+3), f"{name}\n@{rot}", fill=(0,0,0,255), font=font, spacing=0)
    canvas.save(out_path); return out_path

def validate_rules(variants, rs):
    problems = []
    for v in variants:
        if not rs.right_of[v]: problems.append(f"{v} has no RIGHT options")
        if not rs.below_of[v]: problems.append(f"{v} has no DOWN options")
    return problems

def main():
    ap = argparse.ArgumentParser(description="XML-driven rule generator (WFC-like)")
    ap.add_argument("--xml", default="images/samples/grassmud/data.xml", help="Path to XML file")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--height", type=int, default=24)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--tile-dir", default="images/samples/grassmud", help="Folder with <tile>.png images")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--auto-rotate", dest="auto_rotate", action="store_true", default=True,
                    help="Enable rotation inference from declared neighbors (default)")
    ap.add_argument("--no-auto-rotate", dest="auto_rotate", action="store_false",
                    help="Disable rotation inference")
    ap.add_argument("--no-reciprocal", action="store_true")
    args = ap.parse_args()

    tile_defs, variants, explicit_lr, explicit_tb = load_tileset(args.xml)
    rs = compile_rules(tile_defs, variants, explicit_lr, explicit_tb,
                       auto_rotate=args.auto_rotate,
                       auto_reciprocal=not args.no_reciprocal)

    # Optional: quick sanity check
    probs = validate_rules(variants, rs)
    if probs:
        print("[rule-check] potential issues (first 15):")
        for p in probs[:15]: print("  -", p)

    grid = CSPGrid(args.width, args.height, variants, tile_defs, rs, seed=args.seed)
    if not grid.solve():
        raise SystemExit("Failed to solve with current rules. Try loosening constraints or enabling auto-rotate/reciprocal.")

    # Choose an image dir:
    tile_dir = args.tile_dir
    if not tile_dir:
        # default: sibling folder next to XML, e.g. <xml_dir>/tiles
        xml_dir = os.path.dirname(os.path.abspath(args.xml))
        cand = os.path.join(xml_dir, "tiles")
        tile_dir = cand if os.path.isdir(cand) else None

    imgs = load_tile_images(tile_defs, tile_dir)
    outp = render_grid(grid.grid, tile_defs, imgs, args.out)
    layout_json = os.path.splitext(args.out)[0] + ".json"
    with open(layout_json, "w", encoding="utf-8") as f:
        json.dump(grid.grid, f, indent=2)
    print("Saved:", outp, "and", layout_json)

if __name__ == "__main__":
    main()
