"""Minimal reproduction for ibis CacheHandler._finalize_cached_table KeyError.

ibis operation nodes use structural equality — two distinct instances with
identical args compare equal and hash identically.  The CacheHandler uses
these ops as dict keys in _cache_op_to_entry.  When a second structurally-equal
expression is cached after the first's weak ref dies, the dict slot is
overwritten.  The first finalizer then pops the slot; the second hits KeyError.

To see the bug, revert the fix (change `pop(entry.orig_op, None)` back to
`pop(entry.orig_op)` on line 947 of ibis/backends/__init__.py) and run:

    python repro_cache_keyerror.py
"""

import gc

import ibis


def main():
    con = ibis.duckdb.connect()
    t = con.create_table("t", {"x": [1, 2, 3]})

    expr1 = t.mutate(y=ibis.literal(1))
    expr2 = t.mutate(y=ibis.literal(1))

    assert expr1.op() == expr2.op(), "ops must be structurally equal"
    assert expr1.op() is not expr2.op(), "ops must be distinct objects"

    # Cache the first expression
    cached1 = expr1.cache()
    name1 = cached1.op().name

    # Detach the finalizer so the entry stays in the dicts after cached_op is GC'd
    entry1 = con._cache_name_to_entry[name1]
    entry1.finalizer.detach()

    # Drop the only strong reference → cached_op is collected, weak ref dies
    del cached1
    gc.collect()

    # Cache the second (structurally-equal) expression.
    # _cached_table sees entry1 but its weak ref is dead, so it creates entry2
    # and overwrites _cache_op_to_entry[op] — but name1→entry1 is still in
    # _cache_name_to_entry.
    cached2 = expr2.cache()
    name2 = cached2.op().name
    assert name1 != name2, "each cache call should produce a unique name"

    # Finalise name1: pops entry1 from _cache_name_to_entry, then pops the
    # shared op slot from _cache_op_to_entry (actually removing entry2's slot).
    con._finalize_cached_table(name1)

    # Finalise name2: pops entry2 from _cache_name_to_entry, then tries to pop
    # the op slot — already gone → KeyError (without the fix).
    try:
        cached2.release()
    except KeyError as exc:
        print(f"BUG REPRODUCED: KeyError({exc})")
        raise SystemExit(1)

    print("OK — no KeyError, fix is in place")


if __name__ == "__main__":
    main()
