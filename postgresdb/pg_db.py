import dataclasses
import datetime
import functools
import inspect
import logging
import os
import subprocess
import textwrap
from enum import Enum
from itertools import chain
from typing import (Any, Collection, Container, Dict, List, Optional, Tuple,
                    Union)

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from betterpathlib import Path
from dotenv import load_dotenv
from psycopg2 import sql
from psycopg2.errors import UndefinedTable
from psycopg2.extensions import connection, cursor
from psycopg2.extras import DictCursor, Json, execute_batch
from tabulate import tabulate

from postgresdb.utils.itertoolz import (chunks, filter_di_vals, lmap,
                                        sub_dict_inv, valuemap_dict)
from postgresdb.utils.misc import confirm_action, git_root
from postgresdb.utils.parse import blank_str_to_nan, nan_to_none

PathOrStr = Path | str


def psycopg2_nan_to_null(
    f,
    _NULL=psycopg2.extensions.AsIs("NULL"),
    _NaN=np.NaN,
    _Float=psycopg2.extensions.Float,
):
    """Auto convert NaN values to NULL upon insertion"""
    if not pd.isna(f):
        return _Float(f)
    return _NULL


psycopg2.extensions.register_adapter(float, psycopg2_nan_to_null)
psycopg2.extensions.register_adapter(dict, Json)
psycopg2.extensions.register_adapter(np.int64, psycopg2.extensions.AsIs)
psycopg2.extensions.register_adapter(np.bool_, psycopg2.extensions.AsIs)

SqlConstant = Enum("SqlConstant", "NOW DEFAULT")


class PgDb:
    conns = {
        "local": {
            "host": os.environ.get("DB_HOST"),
            "dbname": os.environ.get("DB_NAME"),
            "port": os.environ.get("DB_PORT"),
            "user": os.environ.get("DB_USER"),
            "password": os.environ.get("DB_PASSWORD"),
        }
    }

    def __init__(
        self, conn_name: Optional[str] = None, **connection_params: str
    ) -> None:
        """Initialize database connection.

        Args:
            conn_name: Name of predefined connection from self.conns
            **connection_params: Direct connection parameters (host, dbname, etc.)
        """
        load_dotenv()
        self.conn_key = ""

        # Determine connection parameters
        if not conn_name and not connection_params:
            connection_params = self.conns["local"]
            self.conn_key = "local"
        elif conn_name and conn_name in self.conns:
            connection_params = self.conns[conn_name]
            self.conn_key = conn_name

        if not connection_params or not connection_params.get("host"):
            raise ValueError("No connection params")

        # Establish connection
        self.conn: connection = psycopg2.connect(**connection_params)
        self.cursor: cursor = self.conn.cursor()
        self.dict_cursor: DictCursor = self.conn.cursor(
            cursor_factory=psycopg2.extras.DictCursor
        )

        # Store connection info
        conn_info = self.conn.info
        self.host: str = conn_info.host
        self.port: int = int(conn_info.port)
        self.dbname: str = conn_info.dbname
        self.user: str = conn_info.user
        self.password: str = conn_info.password

        logging.info(f"Connected to database {self.dbname} at host {self.host}")

        # Enable read-only mode for AWS main connections
        if "main" in self.conn_key or "prod" in self.conn_key:
            self.set_readonly()
            logging.info("Auto-enabling read-only mode")

    def __repr__(self) -> str:
        """Return string representation of database connection."""
        if self.conn_key:
            return f"{self.__class__.__name__}({repr(self.conn_key)})"

        return (
            f"{self.__class__.__name__}("
            f"host={repr(self.host)}, "
            f"port={repr(self.port)}, "
            f"dbname={repr(self.dbname)}, "
            f"user={repr(self.user)}, "
            f"password={repr(self.password)})"
        )

    def reopen(self) -> None:
        """Reconnect after close"""
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )
        self.cursor = self.conn.cursor()
        self.di_cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def close(self) -> None:
        self.cursor.close()
        self.conn.close()

    @property
    def readonly(self) -> bool:
        """Return True if connection is in read-only mode."""
        if self.conn.readonly is None:
            return False
        return self.conn.readonly

    def as_readonly(self, value: bool = True) -> "PgDb":
        self.conn.commit()
        self.conn.readonly = value
        return self

    def set_readonly(self, value: bool = True) -> None:
        self.conn.commit()
        self.conn.readonly = value

    def _check_readonly(func) -> Any:  # pylint: disable=no-self-argument
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            if self.conn.readonly:
                raise PermissionError("Database is readonly")
            return func(self, *args, **kwargs)  # pylint: disable=not-callable

        return inner

    def create(self) -> None:
        raise NotImplementedError

    def reset(self, tables, types=()) -> None:
        """Drop the given tables and types; cascading changes"""
        for ents, typ in ((tables, "TABLE"), (types, "TYPE")):
            for entity in ents:
                try:
                    self.cursor.execute(f"DROP {typ} {entity} CASCADE")
                    self.conn.commit()
                except psycopg2.ProgrammingError:
                    self.rollback()
                    self.create()

    @_check_readonly
    def drop_rows(self, table: str) -> None:
        """Drop all rows in `table`"""
        self.cursor.execute(f"TRUNCATE {table} CASCADE")
        self.conn.commit()

    @_check_readonly
    def drop_all(self, schema="public") -> None:
        """Drop all tables and types"""
        if not confirm_action("Delete schema?"):
            return
        self.cursor.execute(
            f"""
            DROP SCHEMA {schema} CASCADE;
            CREATE SCHEMA {schema};
            GRANT ALL ON SCHEMA {schema} TO postgres;
            GRANT ALL ON SCHEMA {schema} TO {schema};
            """
        )

    def rollback(self) -> None:
        """
        Use if you execute a faulty statement and get:
        'InternalError: current transaction is aborted,
        commands ignored until end of transaction block'
        """
        self.cursor.execute("rollback")

    def fetch_val(self, statement: str = "", vals=None) -> Any:
        if statement:
            statement = statement.replace('"', "'")
            self.cursor.execute(statement, vals)
        res = self.cursor.fetchone()
        if res:
            return res[0]
        return None

    def fetch_all(self, statement, vals=None) -> List:
        if statement:
            self.cursor.execute(statement, vals)
        return self.cursor.fetchall()

    def fetch_count(self, statement: str = "", vals=None) -> List:
        """Return the number of values returned"""
        if statement:
            self.cursor.execute(statement, vals)
        return len(self.cursor.fetchall())

    def fetch_column(self, table: str, col: str, where: str = "", vals=None) -> List:
        """Return a column from a table"""
        if where:
            where = "WHERE " + where
        self.cursor.execute(f"SELECT {col} FROM {table} " + where, vals)
        return [t[0] for t in self.cursor.fetchall()]

    def fetch_di(self, statement: str = "", vals=None) -> List[Dict]:
        if statement:
            statement = statement.replace('"', "'")
            self.di_cursor.execute(statement, vals)
        return [dict(row) for row in self.di_cursor.fetchall()]

    def fetch_df(self, statement: str = "", vals=None) -> Optional[pd.DataFrame]:
        """Fetch all results from di_cursor and put them in a DataFrame"""
        if statement:
            statement = statement.replace('"', "'")
            self.di_cursor.execute(statement, vals)
        res = self.di_cursor.fetchall()
        if res:
            res = list(res)
            cols = list(res[0].keys())
            df = pd.DataFrame(res, columns=cols)
            blank_str_to_nan(df)
            df = df.convert_dtypes()
            return df
        return None

    def fetch_mapping(
        self, table: str, key_col: str, value_col: str, where: str = "", vals=None
    ) -> Dict:
        """Return {key_col: value_col} mapping. If multiple rows have the same value in
        the key columns, then the value in the mapping will correspond to that of the last
        of those rows."""
        if where:
            where = "WHERE " + where
        self.di_cursor.execute(
            f"SELECT {key_col}, {value_col} FROM {table} " + where, vals
        )
        return {di[key_col]: di[value_col] for di in self.di_cursor.fetchall()}

    def sample(self, table: str, n_rows: int, columns: List[str] = None) -> List[Dict]:
        """
        Return `n_rows` randomly selected rows from the given `table`,
        optionally restricted to a subset of `columns`.
        """
        select = "*"
        if columns:
            if isinstance(columns, str):
                columns = [columns]
                select = ", ".join(columns)
                self.di_cursor.execute(
                    f"SELECT {select} FROM {table} ORDER BY random() LIMIT %s",
                    (n_rows,),
                )
        return list(self.di_cursor.fetchall())

    @_check_readonly
    def insert_dict(
        self,
        di: Dict,
        table: str,
        return_id=True,
        on_conflict_ignore=False,
        print_statement=False,
        replace_nan_with_none=True,
    ) -> Optional[int]:
        res = self.insert_dicts(
            [di],
            table,
            return_id=return_id,
            on_conflict_ignore=on_conflict_ignore,
            print_statement=print_statement,
            replace_nan_with_none=replace_nan_with_none,
        )
        if return_id:
            return res[0]
        return None

    @_check_readonly
    def insert_dicts(
        self,
        dicts: Union[List[Dict], Tuple[Dict]],
        table: str,
        return_id=True,
        return_cols: Optional[Union[str, Collection[str]]] = None,
        on_conflict_ignore=False,
        print_statement=False,
        replace_nan_with_none=True,
    ) -> List[int]:
        """
        Insert a collection of dictionaries into `table`.

        return_id: if True, return a list of the first primary key of the inserted rows,
          otherwise, return an empty list.
        """
        if not dicts:
            return []
        if replace_nan_with_none:
            dicts = lmap(valuemap_dict(nan_to_none), dicts)
        # if avoid_none_insertion:
        #     dicts = [{k:v for k,v in di.items() if v is not None} for di in dicts]

        if return_id and not return_cols:
            id_cols = self.get_primary_keys(table)
            if not id_cols:
                raise ValueError(
                    "Table does not have a primary key and `return_id` is True"
                )
            return_cols = id_cols[:1]

        # Escape column names with "" in case they match SQL reserved keywords
        cols = dicts[0]
        arg_keys = [f'"{k}"' for k in cols]
        args_str = ", ".join(arg_keys)

        # if not now_fields:
        #     time_fields = self.get_columns_info(table, ['timestamp without time zone', 'timestamp with time zone'])
        #     time_fields = filter_di_vals(lambda di: not di['is_nullable'], time_fields)
        #     now_fields = time_fields.keys() - set(cols)

        vals_str = ", ".join(
            [
                "("
                + ", ".join(["DEFAULT" if v is None else "%s" for v in di.values()])
                + ")"
                for di in dicts
            ]
        )
        statement = f"""
        INSERT INTO {table}
        ({args_str})
        VALUES
        {vals_str}
        """
        if on_conflict_ignore:
            statement += "ON CONFLICT DO NOTHING"
        if return_id:
            return_s = ", ".join(return_cols)
            statement += f"\nRETURNING {return_s}"

        dicts = [{k: v for k, v in di.items() if v is not None} for di in dicts]
        vals = tuple(chain.from_iterable(di.values() for di in dicts))
        if print_statement:
            print(statement)
        self.cursor.execute(statement, vals)
        self.conn.commit()

        if return_id:
            res = self.cursor.fetchall()
            if len(return_cols) == 1:
                res = [t[0] for t in res]
            return res
        return []

    @_check_readonly
    def insert_dataframe(
        self,
        df,
        table: str,
        return_id=True,
        batch_size=5000,
    ) -> List[int]:
        dis: List[Dict] = df.to_dict("records")
        n_inserted = 0
        ids = []
        for dis_batch in chunks(dis, batch_size):
            ids.extend(self.insert_dicts(dis_batch, table, return_id=return_id))
            print("Inserted", len(dis_batch), "rows")
            n_inserted += len(dis_batch)

        print(f"Inserted {n_inserted} company rows into the database")
        return ids

    @_check_readonly
    def insert_dataclass(
        self,
        dclass,
        table: str,
        return_id=True,
    ) -> Optional[int]:
        return self.insert_dicts(dataclasses.asdict(dclass), table, return_id)

    @_check_readonly
    def insert_dataclasses(
        self,
        dclasses,
        table: str,
        return_id=True,
    ) -> List[int]:
        return self.insert_dicts(
            [dataclasses.asdict(p) for p in dclasses], table, return_id
        )

    @_check_readonly
    def update_dicts(
        self,
        dicts: Union[List[Dict], Tuple[Dict]],
        table: str,
        on: str = "id",
        print_statement=False,
        replace_nan_with_none=True,
        update_func=None,
        updated_at_col="updated_at",
    ) -> List[int]:
        """
        Update rows in `table`.

        return_id: if True, return a list of the first primary key of the inserted rows,
                   otherwise, return an empty list.
        update_func: one of
            None
            only_fill: Fill in for missing values only
        """
        if update_func not in (None, "only_fill", "concat"):
            raise ValueError(update_func)
        if not dicts:
            return []
        if replace_nan_with_none:
            dicts = lmap(valuemap_dict(nan_to_none), dicts)
        dicts = tuple(dicts)
        if on not in dicts[0].keys():
            raise ValueError("Could not find update-on column", on, "in dictionary")

        # Escape column names with "" in case they match SQL reserved keywords
        args = [k for k in dicts[0] if k != on]
        # if len(args) > 1 and update_func == 'only_fill':
        #     raise ValueError("Can only replace if updating 1 column")
        if update_func == "only_fill":
            args_str = ",\n".join(
                map(lambda k: f'"{k}" = COALESCE(NULLIF("{k}", \' \'), %({k})s)', args)
            )
        elif update_func == "concat":
            args_str = ",\n".join(
                map(lambda k: f'"{k}" = CONCAT("{k}", %({k})s)', args)
            )
        else:
            args_str = ",\n".join(map(lambda k: f'"{k}" = %({k})s', args))

        if updated_at_col:
            args_str = args_str + f',\n"{updated_at_col}" = NOW()'
        where_str = f'"{on}" = %({on})s'
        statement = f"""
        UPDATE {table}
        SET
        {args_str}
        WHERE
        {where_str}
        """

        if print_statement:
            print(statement)
        execute_batch(self.cursor, statement, dicts)
        self.conn.commit()

    def row_count(self, table: str) -> int:
        """Return the number of rows in the given table"""
        try:
            self.cursor.execute(f"SELECT count(*) FROM {table}")
            return self.cursor.fetchone()[0]
        except UndefinedTable as e:
            self.rollback()
            raise e

    def col_counts(self, table: str, col: str) -> None:
        """Print the number of rows and null-like rows in the given table and columns"""
        rc = self.row_count(table)
        count = self.fetch_val(f"SELECT count({col}) FROM {table}")
        distinct = self.fetch_val(
            f"""
        SELECT COUNT(*) FROM (SELECT DISTINCT {col} FROM {table}) AS temp
        """
        )
        dty = self.get_columns_info(table)[col]["data_type"]
        null_like = ""
        if dty.lower() in ("character varying", "text"):
            null_like = self.fetch_val(f"SELECT count(*) FROM {table} WHERE {col} = ''")
            null_like = f"; {null_like} empty strings"
        elif "array" in dty.lower():
            null_like = self.fetch_val(
                f"SELECT count(*) FROM {table} WHERE caridinality{col} = 0"
            )
            null_like = f"; {null_like} empty arrays"
        print(
            f"{rc} rows ({dty}), of which {count} non-null; {distinct} distinct; {rc-count} null{null_like}"
        )

    def row_counts(self) -> Dict[str, int]:
        """Return the number of rows of all tables"""
        counts = {}
        for tbl in self.list_tables():
            counts[tbl] = self.row_count(tbl)
        return counts

    def get_table_info(self, table: str) -> List[Dict]:
        """
        Get information about a table.
        """
        self.di_cursor.execute(
            """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name=%s
        ORDER BY ordinal_position
        """,
            (table,),
        )
        tbl_schema = self.di_cursor.fetchall()
        return tbl_schema

    def get_table_info2(self, table) -> list:
        # Get the column names
        self.cursor.execute(
            f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table}';
        """
        )
        columns = self.cursor.fetchall()

        # Prepare the results list
        results = []

        # Fetch the first non-null value for each column
        for column in columns:
            column_name = column[0]
            query = sql.SQL(
                "SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 1"
            ).format(column=sql.Identifier(column_name), table=sql.Identifier(table))
            self.cursor.execute(query)
            first_non_null_value = self.cursor.fetchone()
            first_non_null_value = (
                first_non_null_value[0] if first_non_null_value else None
            )
            results.append((column_name, column[1], column[2], first_non_null_value))

        table_schema = pd.DataFrame(
            results,
            columns=["column_name", "data_type", "is_nullable", "example_value"],
        )
        return table_schema

    def get_columns_info(
        self, table: str, data_types: Union[str, Collection[str]] = ()
    ) -> Dict[str, Dict]:
        """
        data_types: on retrieve columns with the given data type(s)
        """
        if isinstance(data_types, str):
            data_types = [data_types]
        tbl_schema = self.get_table_info(table)
        infos = {c["column_name"]: sub_dict_inv(c, "column_name") for c in tbl_schema}
        for info in infos.values():
            info["is_nullable"] = info["is_nullable"] == "YES"
        if data_types:
            infos = filter_di_vals(lambda di: di["data_type"] in data_types, infos)

        return infos

    def describe_table(
        self,
        tables: Union[str, List[str]] = "",
        with_example=False,
        schema: str = "public",
    ) -> None:
        """
        Print information about a table, or about all tables if no table is specified.
        """
        if not tables:
            tables = self.list_tables(schema)
        elif isinstance(tables, str):
            tables = [tables]
        headers = ["col", "type", "nullable", "default"]
        if with_example:
            headers.append("example")

        for tbl in tables:
            rc = self.row_count(tbl)
            print("\n", tbl.capitalize(), f"(row count: {rc})")
            rows = self.get_table_info(tbl)

            # Selecting tables dynamically is not supported by the regular (above) approach.
            if with_example:
                self.cursor.execute(
                    sql.SQL("SELECT * FROM {} LIMIT 1").format(sql.Identifier(tbl))
                )
                fst_row = self.cursor.fetchone()
                if not fst_row:
                    fst_row = [None] * len(rows)
                rows = [col + [el] for col, el in zip(rows, fst_row)]
            print(tabulate(rows, headers))

    def describe_col(self, table: str, column: str, with_example=True) -> None:
        """
        Print information about a table, or about all tables if no table is specified.
        """
        headers = ["col", "type", "nullable", "default"]
        if with_example:
            headers.append("example")

        rows = self.get_table_info(table)
        rows = [r for r in rows if r[0] == column]

        # Selecting tables dynamically is not supported by the regular (above) approach.
        if with_example:
            self.cursor.execute(
                sql.SQL("SELECT * FROM {} LIMIT 1").format(sql.Identifier(table))
            )
            fst_row = self.cursor.fetchone()
            if not fst_row:
                fst_row = [None] * len(rows)
            rows = [col + [el] for col, el in zip(rows, fst_row)]
        print(tabulate(rows, headers))

    def list_tables(self, schema="public") -> List[str]:
        """Print the name of all the tables in the database"""
        self.cursor.execute(
            sql.SQL(
                """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = {}"""
            ).format(sql.Literal(schema))
        )
        tables = [x[0] for x in self.cursor.fetchall()]
        return tables

    def table_constraints(self, table, schema="public") -> List[Dict]:
        self.di_cursor.execute(
            f"""
        SELECT con.*
        FROM pg_catalog.pg_constraint con
             INNER JOIN pg_catalog.pg_class rel
                        ON rel.oid = con.conrelid
             INNER JOIN pg_catalog.pg_namespace nsp
                        ON nsp.oid = connamespace
        WHERE nsp.nspname = '{schema}'
              AND rel.relname = '{table}';
        """
        )
        return self.di_cursor.fetchall()

    def get_column_names(self, table) -> List[str]:
        self.cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name=%s
            """,
            (table,),
        )
        column_names = [r[0] for r in self.cursor.fetchall()]
        return column_names

    def insert_statement(
        self, table: str, exclude: Container = (), exclude_id: bool = True
    ) -> str:
        id_col = None

        column_names = [r for r in self.get_column_names(table) if r not in exclude]
        if table + "_id" in column_names:
            id_col = table + "_id"
        elif table[:-1] + "_id" in column_names:
            id_col = table[:-1] + "_id"
        elif "id" in column_names:
            id_col = "id"
        if exclude_id:
            column_names -= set([id_col])
        args = textwrap.fill(", ".join(column_names), 80)
        vals = textwrap.fill(", ".join(f"%({arg})s" for arg in column_names), 80)
        statement = f"INSERT INTO {table}\n({args})\nVALUES\n({vals})"
        return statement

    def get_primary_keys(self, table: str) -> List[str]:
        """
        Get a list of primary keys in `table`
        """
        self.cursor.execute(
            f"""
        SELECT a.attname
        FROM   pg_index i
        JOIN   pg_attribute a ON a.attrelid = i.indrelid
                              AND a.attnum = ANY(i.indkey)
        WHERE  i.indrelid = '{table}'::regclass
        AND    i.indisprimary """
        )
        return [t[0] for t in self.cursor.fetchall()]

    def get_last_inserted(self, table: str, n: int = 10) -> List[Dict]:
        """
        Return the `n` entries from `table` with the largest primary key.
        """
        id_col = self.get_primary_keys(table)[0]
        self.di_cursor.execute(
            f"SELECT * from {table} ORDER BY {id_col} DESC LIMIT {n}",
        )
        return self.di_cursor.fetchall()

    def n_active_connections(self) -> int:
        """Return the number of active connections, including this one"""
        self.cursor.execute(
            "SELECT COUNT(*) FROM pg_stat_activity WHERE datname = %s", (self.dbname,)
        )
        self.conn.commit()
        return self.cursor.fetchone()[0]

    @staticmethod
    def format_identifiers(statement, *args, **kwargs) -> psycopg2.sql.Composed:
        args = [sql.Identifier(arg) for arg in args]
        kwargs = valuemap_dict(sql.Identifier, kwargs)
        sql_st = sql.SQL(statement).format(*args, **kwargs)
        return sql_st

    @_check_readonly
    def rename_column(self, table: str, from_col: str, to_col: str) -> None:
        seql = self.format_identifiers(
            "ALTER TABLE {} RENAME COLUMN {} TO {}", table, from_col, to_col
        )
        self.cursor.execute(seql)
        self.conn.commit()

    @_check_readonly
    def rename_table(self, from_table: str, to_table: str) -> None:
        seql = self.format_identifiers(
            "ALTER TABLE {} RENAME TO {}", from_table, to_table
        )
        self.cursor.execute(seql)
        self.conn.commit()

    @_check_readonly
    def dedupe_column(self, table, col, id_col="id") -> None:
        n_dupes = len(
            self.fetch_all(
                f"""
        DELETE FROM
            {table} a
            USING {table} b
        WHERE
            a.{id_col} < b.{id_col}
            AND a.{col} = b.{col}
        RETURNING 1
        """
            )
        )
        self.conn.commit()
        print(f"Dropped {n_dupes} duplicates for {table}:{col}")

    @_check_readonly
    def dedupe_columns(self, table, cols: Collection[str], id_col) -> pd.DataFrame:
        cols_s = ", ".join(cols)
        df = self.fetch_df(
            """
        DELETE FROM {table}
        WHERE {table}.{id_col} NOT IN
        (SELECT {id_col} FROM (
            SELECT DISTINCT ON ({cols_s}) *
        FROM {table}) AS unneeded_alias)
        RETURNING *
        """
        )
        return df

    def value_counts(self, table, column) -> pd.DataFrame:
        return self.fetch_df(
            f"""
        SELECT {column}, count(*)
        FROM {table}
        GROUP BY {column}
        """
        )

    def backup_to_file(
        self,
        db_path: PathOrStr,
        desc="",
        next_fname=True,
        bin_path="pg_dump",
    ) -> Path:
        """
        Re
        """
        date = str(datetime.datetime.now().date())
        if desc:
            desc = "-" + desc
        default_fname = f"{self.dbname}-{date}{desc}.sql.c"
        db_path = Path(db_path)
        if db_path.is_dir():
            db_path = db_path / default_fname
        if next_fname:
            db_path = db_path.next_unused_path()
        if not db_path.exists():
            db_path = git_root() / "db_backup" / db_path
        db_path = str(db_path)
        env = dict(os.environ)
        env["PGPASSWORD"] = self.password
        cmd = (
            f"{bin_path} -d {self.dbname} -h {self.host} -U {self.user} "
            f"--no-owner --no-privileges --format c -f {db_path}"
        )
        print(cmd)
        subprocess.run(cmd, shell=True, env=env)
        print(f"Backed up database {self.dbname} at host {self.host} to {db_path}")
        return Path(db_path)

    @_check_readonly
    def restore_from_file(
        self,
        db_path: PathOrStr,
        n_concurrent: int = 4,
        bin_path="pg_restore",
        verbose=False,
    ) -> None:
        """
        Temporily closes the connection in order for the restoration to proceed.
        If `db_path` is not given, will choose most recently modified file.

        If restoration fails, try `n_concurrent=1`.
        """
        n_conns = self.n_active_connections()
        if n_conns > 1:
            print(
                f"WARNING: Number of active DB connections, including self is {n_conns}, "
                "restoration will probably not start until other connections are closed."
            )
        if "main" in self.conn_key or "prod" in self.conn_key:
            if not confirm_action("Really restore production database?"):
                return
        self.close()

        db_path = Path(db_path)
        if not db_path.exists():
            db_path = git_root() / "db_backup" / db_path
        if not db_path.exists():
            raise ValueError(f"Path {db_path} does not exist")
        env = dict(os.environ)
        env["PGPASSWORD"] = self.password
        verbose = "-v" if verbose else ""
        cmd = (
            f"{bin_path} -j {n_concurrent} -d {self.dbname} -h {self.host} -U {self.user} "
            f"--clean --no-owner --no-privileges {db_path} {verbose}"
        )
        print(cmd)
        subprocess.run(cmd, shell=True, env=env)
        print(f"Restored database {self.dbname} at host {self.host} from {db_path}")

        self.reopen()


postgres_types = {
    bool: "BOOLEAN",
    str: "TEXT",
    datetime.datetime: "TIMESTAMPTZ",
    int: "INTEGER",
}


def table_from_dataclass(dataclass):
    """Generate a `CREATE TABLE ...` statement given a dataclass instance or class"""
    name = dataclass.__name__
    cols = []
    for field in dataclasses.fields(dataclass):
        not_null = "NOT NULL"
        typ = field.type
        if (
            hasattr(typ, "__args__") and type(None) in typ.__args__
        ):  # pylint: disable=unidiomatic-typecheck
            # Field is optional
            not_null = ""
            typ = (
                typ.__args__[0]
                if isinstance(typ.__args__[1], type(None))
                else typ.__args__[1]
            )
        default = "" if field.default is dataclasses.MISSING else str(field.default)
        sql_typ = postgres_types[typ]
        cols.append(f"{field.name:<20}{sql_typ:<12}{default} {not_null:<10}")
    return f"CREATE TABLE {name}\n" "" + "\n".join(cols)


def db_or_conn(args: Union[str, Collection[str]] = "db", db_class=Db):
    """
    A decorater that automatically opens and closes a database if the value of `arg`
    is a connection string or dict, else just passes along the database as is.
    """
    if isinstance(args, str):
        args = [args]

    def deco(func):
        sig = inspect.signature(func)
        for arg in args:
            if arg not in sig.parameters:
                raise ValueError(f"{arg} not a parameter of decorated function")

        @functools.wraps(func)
        def inner(*_args, **kwargs):
            bind = sig.bind_partial(*_args, **kwargs)
            bind.apply_defaults()
            opened = [False] * len(args)
            for i, arg in enumerate(args):
                if not isinstance(bind.arguments[arg], Db):
                    db = db_class(bind.arguments[arg])
                    bind.arguments[arg] = db
                    opened[i] = True
            result = func(**bind.arguments)
            for i, arg in enumerate(args):
                if opened[i]:
                    bind.arguments[arg].close()
            return result

        return inner

    return deco
