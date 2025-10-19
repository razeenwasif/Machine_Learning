""" Module with functionality to save the linkage results to a CSV file.
"""

# =============================================================================
# Import necessary modules

import csv
import os

# -----------------------------------------------------------------------------

def save_linkage_set(file_name, class_match_set):
  """Write the given set of matches (record pair identifiers) into a CSV file
     with one pair of record identifiers per line)

     Parameter Description:
       file_name       : Name of the data file to be write into (a CSV file)
       class_match_set : The set of classified matches (pairs of record
                         identifiers) 
  """
  print('Write linkage results to file: ' + file_name)
  
  # Create the directory if it does not exist
  os.makedirs(os.path.dirname(file_name), exist_ok=True)

  with open(file_name, 'w') as csv_file:  # Open a CSV file for writing
    csv_writer = csv.writer(csv_file)
    for (rec_id1, rec_id2) in sorted(class_match_set):  # Sort for nicer output
      csv_writer.writerow([rec_id1, rec_id2])

  print('  Wrote %d linked record pairs' % (len(class_match_set)))
  print('')

# -----------------------------------------------------------------------------

# End of program.


