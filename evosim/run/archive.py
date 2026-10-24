"""Genome archive: notable genomes deduplicated by hash, stored in SQLite with a size cap.

Selected every N ticks: most abundant genome, genome of the largest organism, longest-lived lineage,
most distinct `state` values used (within one organism), deepest division, and a small random sample.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np

from ..genome import assembler, isa

SCHEMA = """
CREATE TABLE IF NOT EXISTS genomes(
  hash TEXT PRIMARY KEY, words BLOB, length INTEGER, first_tick INTEGER, last_tick INTEGER,
  peak_count INTEGER, lineage INTEGER, parent_hash TEXT, trait_means TEXT, tags TEXT, added_wall REAL,
  source TEXT);
CREATE INDEX IF NOT EXISTS idx_tick ON genomes(first_tick);
"""
TAG_PRIORITY = ("most_abundant", "largest_organism", "longest_lineage", "most_states", "deepest", "random", "manual")


class GenomeArchive:
    def __init__(self, path: str, max_bytes: int = 64 * 1024 * 1024):
        self.path = path
        self.max_bytes = max_bytes
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.close()

    # ------------------------------------------------------------------ writing
    def add(self, words, first_tick, count, lineage, parent_hash=None, traits=None, tag="manual", source="run"):
        h = f"{assembler.genome_hash(words):016x}"
        arr = np.asarray([w & 0xFFFFFFFF for w in words], np.uint32).tobytes()
        row = self.db.execute("SELECT peak_count, tags FROM genomes WHERE hash=?", (h,)).fetchone()
        if row:
            tags = set(json.loads(row[1])) | {tag}
            self.db.execute("UPDATE genomes SET peak_count=?, last_tick=?, tags=? WHERE hash=?",
                            (max(row[0], int(count)), int(first_tick), json.dumps(sorted(tags)), h))
        else:
            self.db.execute("INSERT INTO genomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (h, arr, len(words), int(first_tick), int(first_tick), int(count), int(lineage),
                             parent_hash, json.dumps(traits or {}), json.dumps([tag]), time.time(), source))
        return h

    def add_from_sim(self, sim):
        """Archive the notable genomes of the current population. Returns the number of genomes touched."""
        st = sim.state
        c = st.cells
        al = np.nonzero(c.alive.numpy() != 0)[0]
        if len(al) == 0:
            return 0
        gid = c.genome_id.numpy()[al]
        lin = c.lineage_id.numpy()[al]
        org = c.organism_id.numpy()[al]
        depth = c.division_depth.numpy()[al]
        birth = c.birth_tick.numpy()[al]
        state = c.state.numpy()[al]
        traits = c.traits.numpy()[al]
        g = st.genomes
        glen = g.length.numpy()
        gw = g.words.numpy()
        gpar = g.parent.numpy()
        ghash = g.hash.numpy()
        gft = g.first_tick.numpy()
        ug, gc = np.unique(gid, return_counts=True)
        picks: dict[int, str] = {}
        picks[int(ug[np.argmax(gc)])] = "most_abundant"
        uo, oc = np.unique(org, return_counts=True)
        big = uo[np.argmax(oc)]
        picks.setdefault(int(gid[org == big][0]), "largest_organism")
        ul = np.unique(lin)
        oldest = {int(l): birth[lin == l].min() for l in ul[:2000]}
        picks.setdefault(int(gid[lin == min(oldest, key=oldest.get)][0]), "longest_lineage")
        best, best_o = 0, None
        for o in uo[np.argsort(-oc)[:50]]:
            k = len(np.unique(state[org == o]))
            if k > best:
                best, best_o = k, o
        if best_o is not None and best > 1:
            picks.setdefault(int(gid[org == best_o][0]), "most_states")
        picks.setdefault(int(gid[np.argmax(depth)]), "deepest")
        rs = np.random.default_rng(int(sim.tick))
        for q in rs.choice(len(al), size=min(3, len(al)), replace=False):
            picks.setdefault(int(gid[q]), "random")
        n = 0
        for gi, tag in picks.items():
            words = [int(w) & 0xFFFFFFFF for w in gw[gi, :glen[gi]]]
            sel = gid == gi
            tm = {nme: float(traits[sel, t].mean()) for t, nme in enumerate(isa.TRAIT_NAMES)}
            ph = gpar[gi]
            parent_hash = f"{int(ghash[ph]) & (2**64 - 1):016x}" if ph >= 0 else None
            self.add(words, int(gft[gi]), int(sel.sum()), int(lin[sel][0]), parent_hash, tm, tag)
            n += 1
        self.db.commit()
        self.prune()
        return n

    def prune(self):
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        if size <= self.max_bytes:
            return 0
        removed = 0
        rows = self.db.execute("SELECT hash, tags, peak_count FROM genomes ORDER BY added_wall ASC").fetchall()
        for h, tags, peak in rows:
            if size <= self.max_bytes * 0.9:
                break
            t = json.loads(tags)
            if "manual" in t or "most_abundant" in t and peak > 1000:
                continue
            self.db.execute("DELETE FROM genomes WHERE hash=?", (h,))
            removed += 1
            size -= 400 + 4 * 64
        self.db.commit()
        self.db.execute("VACUUM")
        return removed

    def import_archive(self, other_path):
        o = sqlite3.connect(other_path)
        rows = o.execute("SELECT hash, words, length, first_tick, last_tick, peak_count, lineage, parent_hash, "
                         "trait_means, tags, added_wall FROM genomes").fetchall()
        for r in rows:
            self.db.execute("INSERT OR IGNORE INTO genomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            r + (f"import:{os.path.basename(os.path.dirname(other_path))}",))
        self.db.commit()
        o.close()
        return len(rows)

    # ------------------------------------------------------------------ reading
    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM genomes").fetchone()[0]

    def list(self, limit=200, tag=None):
        q = "SELECT hash, length, first_tick, peak_count, lineage, parent_hash, tags, trait_means FROM genomes"
        args = ()
        if tag:
            q += " WHERE tags LIKE ?"
            args = (f'%"{tag}"%',)
        q += " ORDER BY peak_count DESC LIMIT ?"
        return self.db.execute(q, args + (limit,)).fetchall()

    def get(self, h):
        r = self.db.execute("SELECT words FROM genomes WHERE hash=?", (h,)).fetchone()
        return [int(x) for x in np.frombuffer(r[0], np.uint32)] if r else None

    def select(self, selection="random", k=1, rs=None):
        """Genome word lists for inject_archive."""
        rs = rs or np.random.default_rng(0)
        order = {"most_abundant": "peak_count DESC", "largest_organism": None, "longest_lived": None,
                 "most_recent": "added_wall DESC", "random": None}[selection]
        tagmap = {"largest_organism": "largest_organism", "longest_lived": "longest_lineage"}
        if selection in tagmap:
            rows = self.db.execute("SELECT hash FROM genomes WHERE tags LIKE ? ORDER BY peak_count DESC",
                                   (f'%"{tagmap[selection]}"%',)).fetchall()
        elif order:
            rows = self.db.execute(f"SELECT hash FROM genomes ORDER BY {order} LIMIT 50").fetchall()
        else:
            rows = self.db.execute("SELECT hash FROM genomes").fetchall()
        if not rows:
            return []
        hs = [r[0] for r in rows]
        if selection == "random":
            hs = [hs[i] for i in rs.integers(0, len(hs), size=k)]
        else:
            hs = [hs[i % len(hs)] for i in range(k)] if len(hs) < k else hs[:max(1, min(k, len(hs)))]
        return [self.get(h) for h in hs]
