from __future__ import annotations

import gc

import pytest
from pytest import mark

import ibis
import ibis.common.exceptions as com

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")

pytestmark = [
    mark.notyet(
        ["databricks"],
        reason="Databricks does not support temporary tables, even though they allow the syntax",
    ),
    mark.notyet(["athena"], reason="Amazon Athena doesn't support temporary tables"),
]


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression(backend, alltypes):
    non_persisted_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_calc=ibis.literal("xyz")
    )
    persisted_table = non_persisted_table.cache()
    backend.assert_frame_equal(
        non_persisted_table.order_by("id").to_pandas(),
        persisted_table.order_by("id").to_pandas(),
    )


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_contextmanager(backend, con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_column=ibis.literal("big calc")
    )
    with non_cached_table.cache() as cached_table:
        backend.assert_frame_equal(
            non_cached_table.order_by("id").to_pandas(),
            cached_table.order_by("id").to_pandas(),
        )
    assert non_cached_table.op() not in con._cache_op_to_entry


@mark.notimpl(["flink", "impala", "trino", "druid"])
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
def test_persist_expression_multiple_refs(backend, con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_column=ibis.literal("big calc 2")
    )
    op = non_cached_table.op()
    cached_table = non_cached_table.cache()

    backend.assert_frame_equal(
        non_cached_table.order_by("id").to_pandas(),
        cached_table.order_by("id").to_pandas(),
        check_dtype=False,
    )

    name = cached_table.op().name
    nested_cached_table = non_cached_table.cache()

    # cached tables are identical and reusing the same op
    assert cached_table.op() is nested_cached_table.op()
    # table is cached
    assert op in con._cache_op_to_entry

    # deleting the first reference, leaves table in cache
    del nested_cached_table
    assert op in con._cache_op_to_entry

    # deleting the last reference, releases table from cache
    del cached_table
    assert op not in con._cache_op_to_entry

    # assert that table has been dropped
    assert name not in con.list_tables()


@mark.notimpl(["flink", "impala", "trino", "druid"])
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_repeated_cache(alltypes, con):
    non_cached_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_column=ibis.literal("big calc 2")
    )
    cached_table = non_cached_table.cache()
    nested_cached_table = cached_table.cache()
    name = cached_table.op().name

    assert not nested_cached_table.to_pandas().empty

    del nested_cached_table, cached_table

    assert name not in con.list_tables()


@mark.notimpl(["flink", "impala", "trino", "druid"])
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_release(con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_column=ibis.literal("big calc 3")
    )
    cached_table = non_cached_table.cache()
    cached_table.release()

    assert non_cached_table.op() not in con._cache_op_to_entry

    # a second release does not hurt
    cached_table.release()

    with pytest.raises(Exception, match=cached_table.op().name):
        cached_table.execute()


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_cache_concurrent_structurally_equal_expressions(con, alltypes):
    expr1 = alltypes.mutate(
        test_column=ibis.literal("calculation"),
        other_column=ibis.literal("structurally equal"),
    )
    expr2 = alltypes.mutate(
        test_column=ibis.literal("calculation"),
        other_column=ibis.literal("structurally equal"),
    )

    assert expr1.op() == expr2.op()
    assert expr1.op() is not expr2.op()

    cached1 = expr1.cache()
    name1 = cached1.op().name

    # Detach the finalizer so the entry stays in the dicts after the cached op
    # is garbage-collected (simulates a race where two requests share a backend).
    entry1 = con._cache_name_to_entry[name1]
    entry1.finalizer.detach()

    del cached1
    gc.collect()

    # entry1's weak ref is now dead; caching expr2 creates a new entry that
    # overwrites the shared _cache_op_to_entry slot.
    cached2 = expr2.cache()
    name2 = cached2.op().name
    assert name1 != name2

    # Finalise the first name — must not evict the newer entry's mapping.
    con._finalize_cached_table(name1)

    entry2 = con._cache_name_to_entry[name2]
    assert con._cache_op_to_entry.get(expr2.op()) is entry2

    # Finalise the second — must not raise KeyError.
    cached2.release()

    assert expr2.op() not in con._cache_op_to_entry
