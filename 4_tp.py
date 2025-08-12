import os
import pyodbc
import pandas as pd

def to_datetime(df, column):

    df[column] = pd.to_datetime(df[column], errors='coerce')
    df.loc[df[column].dt.normalize() == pd.Timestamp('1900-01-01'), column] = pd.NaT

    return df

def to_lower_case(df, column):

	df[column] = df[column].astype(str).str.lower()

	return df

def load_tps():

    connection = pyodbc.connect(
                            'DRIVER={SQL Server};'
                            'SERVER=10.46.9.18;'
                            'DATABASE=dwh;'
                            'UID=marketing_User;'
                            'PWD=Market!23'
                            )
    cursor = connection.cursor()
    select = f"""
    	with
    		account_flt as
    		(
    			select
    				accountpk
    			from
					DwhDimAccount
				where
					createdon between '2024-01-01' and '2025-07-01'
					and
					lower(descwebsite) IN ('fortrade.com', 'gcmforex', 'kapitalrs')
    		)
			select
				tp.accountpk	account_pk,
				tp.tppk			tp_pk,
				tp.tpid			tp_id,
				createdon		tp_created_on,
				demoflag		tp_is_demo,
				broker,
				currency,
				leverage,
				balance
			from
				DwhDimTp tp
			join
				account_flt	af
			on
				tp.accountpk = af.accountpk
    """
    cursor.execute(select)
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    cursor.close()
    connection.close()

    tps = pd.DataFrame.from_records(rows, columns=columns)

    tps = to_datetime(tps, 'tp_created_on')
    tps = to_lower_case(tps, 'currency')
    tps = to_lower_case(tps, 'broker')

    tps = tps.sort_values(by='tp_created_on', ignore_index=True)
    
    return tps

# Main

parquet_file_dir = 'datasources'
parquet_file_name = 'tp.parquet'
parquet_file_path = os.path.join(parquet_file_dir, parquet_file_name)

tps = load_tps()
tps.to_parquet(parquet_file_path, engine="pyarrow", compression="snappy")
