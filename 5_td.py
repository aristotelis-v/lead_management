import os
import pyodbc
import pandas as pd

def load_data(start_date: str, end_date: str) -> pd.DataFrame:

    conn_str_template = (
        'DRIVER={{SQL Server}};'
        'SERVER=10.46.9.18;'
        'DATABASE=dwh;'
        'UID={user};'
        'PWD={pwd};'
    )

    credentials = {
        'user': os.getenv('DB_USER', 'marketing_user'),
        'pwd': os.getenv('DB_PWD', 'Market!23'),
    }

    sql = """
        WITH
            ac_flt AS
            (
                SELECT
                    accountpk
                FROM
                    DwhDimAccount
                WHERE
                    createdon BETWEEN ? AND ?
                    AND
                    lower(descwebsite) IN ('fortrade.com', 'gcmforex', 'kapitalrs')
            ),
            tx_flt AS
            (
                SELECT
                    ticketpk            ticket_pk,
                    tx.accountpk        account_pk,
                    tppk                tp_pk,
                    commandpk           command_pk,
                    symbolpk            symbol_pk,
                    opentimeutc         open_time_utc,
                    realtradeflag       real_trade,
                    profitusd           profit_usd
                FROM
                    DwhFactTransaction  tx
                JOIN
                    ac_flt              af
                ON
                    tx.accountpk = af.accountpk
                WHERE
                    cancelledflag = 0
            ),
            cm_flt AS
            (
                SELECT DISTINCT
                    commandpk           command_pk,
                    lower(categoryname) command_name
                FROM
                    DwhDimCommand
                WHERE
                    lower(categoryname) IN ('buy', 'buy limit', 'buy stop', 'sell', 'sell limit', 'sell stop')
            ),
            sm_flt AS
            (
                SELECT DISTINCT
                    symbolpk            symbol_pk,
                    symbolnewname       symbol_name
                FROM
                    DwhDimSymbol
            )
            SELECT
                ticket_pk,
                account_pk,
                tp_pk,
                tx.command_pk,
                command_name,
                tx.symbol_pk,
                symbol_name,
                open_time_utc,
                real_trade,
                profit_usd
            FROM
                tx_flt tx
            JOIN
                cm_flt cf
            ON
                tx.command_pk = cf.command_pk
            JOIN
                sm_flt sf
            ON
                tx.symbol_pk = sf.symbol_pk
    """

    conn_str = conn_str_template.format(**credentials)

    with pyodbc.connect(conn_str) as conn:

        df = pd.read_sql_query(sql, conn, params=[start_date, end_date])

    return df


if __name__ == "__main__":

    df = load_data('2024-01-01', '2025-07-01')
    print(df.head())
    print(df.shape)

    out_dir = 'datasources'
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(
        os.path.join(out_dir, 'td.parquet'),
        engine='pyarrow',
        compression='snappy'
    )
