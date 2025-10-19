""" Module with functionalities for reading data from a file and return a
    dictionary with record identifiers as keys and a list of attribute values.

    Also provides a function to load a truth data set of record pairs that are
    matches.
"""

# =============================================================================
# Import necessary modules

import cudf
import gzip
import csv

# -----------------------------------------------------------------------------

def load_data_set(file_name, rec_id_col_name, use_attr_list):
    """Load the data set and store in memory as a cuDF DataFrame.

    Parameter Description:
        file_name (str): Name of the data file to be read (CSV or CSV.GZ file).
        rec_id_col_name (str): The name of the record identifier column.
        use_attr_list (list): List of attribute names to extract from the file.
    """

    print(f'Load data set from file: {file_name}')

    # Use cuDF to read the CSV file
    gdf = cudf.read_csv(file_name)

    # Ensure all column names are lowercase for consistency
    gdf.columns = [col.lower() for col in gdf.columns]

    # Identify the record identifier column and attributes to use by name
    rec_id_col = rec_id_col_name.lower()
    use_cols = [rec_id_col] + [col.lower() for col in use_attr_list]

    # Filter the DataFrame to keep only the necessary columns
    gdf = gdf[use_cols]

    # Convert all string columns to lowercase
    for col in gdf.select_dtypes(include=['object']).columns:
        gdf[col] = gdf[col].str.lower()

    # Set the record identifier as the index
    gdf = gdf.set_index(rec_id_col)

    print(f'  Record identifier attribute: {rec_id_col}')
    print('  Attributes to use:')
    print(f"    {' '.join(attr.lower() for attr in use_attr_list)}")

    print(f'  Loaded {len(gdf)} records.')
    print('')

    return gdf

def load_truth_data(file_name):
  """Load a truth data file where each line contains two record identifiers
     where the corresponding record pair is a true match.

     Returns a set where the elements are pairs (tuples) of these record
     identifier pairs.
  """

  if (file_name.endswith('gz')):
    in_f = gzip.open(file_name, 'rt')
  else:
    in_f = open(file_name)

  csv_reader = csv.reader(in_f)

  print('Load truth data from file: '+file_name)

  truth_data_set = set()

  for rec_list in csv_reader:
    assert len(rec_list) == 2, rec_list  # Make sure only two identifiers

    rec_id1 = rec_list[0].lower()
    rec_id2 = rec_list[1].lower()

    truth_data_set.add((rec_id1,rec_id2))

  in_f.close()

  print('  Loaded %d true matching record pairs' % (len(truth_data_set)))
  print('')

  return truth_data_set

# -----------------------------------------------------------------------------

# End of program.
