# A program to generate individual student record linkage data sets from
# provided master data sets.
#
# Important: Must be run under Python 3
#
# Peter Christen, Sept 2020
# Updated by Anushka Vidanage, Aug 2023
# Updated by Charini Nanayakkara, Aug 2024
# Updated by Charini Nanayakkara, Aug 2025
#
# Usage:
#
#  python3 generate-student-datasets-rl.py [anu_id]
#
# -----------------------------------------------------------------------------

import gzip
import hashlib
import random
import sys
import itertools

# Do NOT change these file names!
#
rl_dataset_name1 = 'dw_assignment_master_rl1.csv.gz'
rl_dataset_name2 = 'dw_assignment_master_rl2.csv.gz'
rl_gt_data_name =  'dw_assignment_master_rlgt.csv.gz'

# Check the program is run under Python3
#
version = sys.version_info[0]
if (version != 3):
  print('*** Program must be run using Python 3 ***')
  sys.exit()

anu_id = sys.argv[1].lower()
assert len(anu_id) == 8, 'Invalid ANU ID provided'
assert anu_id[0] == 'u', ('Invalid ANU ID provided', anu_id[0])
assert anu_id[1:].isdigit() == True, 'Invalid ANU ID provided'

salt = 'sem2/2025'

md5_anu_id = hashlib.md5(anu_id.encode('utf-8')).hexdigest()

rl_dataset1 =   gzip.open(rl_dataset_name1, mode='rt')
rl_dataset2 =   gzip.open(rl_dataset_name2, mode='rt')
rl_dataset_gt = gzip.open(rl_gt_data_name,  mode='rt')

rl_headerline1 = rl_dataset1.readline()
rl_headerline2 = rl_dataset2.readline()

rl_dict1 = {}
rl_dict2 = {}

for rec1 in rl_dataset1:
  rec_id1 = rec1.split(',')[0].strip()
  assert rec_id1 not in rl_dict1
  rl_dict1[rec_id1] = rec1
rl_dataset1.close()

for rec2 in rl_dataset2:
  rec_id2 = rec2.split(',')[0].strip()
  assert rec_id2 not in rl_dict2
  rl_dict2[rec_id2] = rec2
rl_dataset2.close()

rl_id_list1 = sorted(rl_dict1.keys())
rl_id_list2 = sorted(rl_dict2.keys())

gt_list = []
gt_dict = {}
for rec_pair in rl_dataset_gt:
  rec_id1, rec_id2 = rec_pair.split(',')
  gt_list.append([rec_pair, rec_id1.strip(), rec_id2.strip()])
  gt_dict[rec_id1.strip()] = rec_id2.strip()

random.seed(anu_id+salt)

student_gt_list = random.sample(gt_list, 10000)

matched_rec_id_set1 = set()
matched_rec_id_set2 = set()

student_data_set_list1 = []
student_data_set_list2 = []
student_gt_tuple_list = []
for [rec_pair, rec_id1, rec_id2] in student_gt_list:
  student_gt_tuple_list.append(rec_pair)
  rec_id1 = rec_id1.strip()
  rec_id2 = rec_id2.strip()
  matched_rec_id_set1.add(rec_id1)
  matched_rec_id_set2.add(rec_id2)
  student_data_set_list1.append(rl_dict1[rec_id1])
  student_data_set_list2.append(rl_dict2[rec_id2])

gt_rec2_list = []

i = 0
while (len(student_data_set_list1) < 20000):
  rec_id1 = rl_id_list1[i]
  if (rec_id1 not in matched_rec_id_set1):
    student_data_set_list1.append(rl_dict1[rec_id1])

    if(rec_id1 in gt_dict.keys()):
      gt_rec2_list.append(gt_dict[rec_id1])
  i += 1

i = 0
while (len(student_data_set_list2) < 20000):
  rec_id2 = rl_id_list2[i]
  if (rec_id2 not in matched_rec_id_set2 and rec_id2 not in gt_rec2_list):
    student_data_set_list2.append(rl_dict2[rec_id2])
  i += 1

random.seed(anu_id+salt)
random.shuffle(student_data_set_list1)
random.shuffle(student_data_set_list2)

md5_str = ''

student_data_set_name1 =   'data_wrangling_rl1_2025_%s.csv' % (anu_id)
student_data_set_name2 =   'data_wrangling_rl2_2025_%s.csv' % (anu_id)
student_gt_data_set_name = 'data_wrangling_rlgt_2025_%s.csv' % (anu_id)

student_file1 = open(student_data_set_name1, 'w')
student_file1.write(rl_headerline1)

for rec in student_data_set_list1:
  student_file1.write(rec)
  md5_str += rec.strip()  # Remove OS depdendent new lines characters
student_file1.close()

student_file2 = open(student_data_set_name2, 'w')
student_file2.write(rl_headerline2)

for rec in student_data_set_list2:
  student_file2.write(rec)
  md5_str += rec.strip()  # Remove OS depdendent new lines characters
student_file2.close()

student_gt_file = open(student_gt_data_set_name, 'w')
for rec_pair in student_gt_tuple_list:
  student_gt_file.write(rec_pair)
  md5_str += rec_pair.strip()  # Remove OS depdendent new lines characters
student_gt_file.close()

md5_data_set = hashlib.md5(md5_str.encode('utf-8')).hexdigest()

print('-'*40)
print()
print('Your student data sets for the third data wrangling 2025 assignment')
print('have been generated and written into files:')
print()
print(' ',student_data_set_name1)
print(' ',student_data_set_name2)
print(' ',student_gt_data_set_name)
print()
print('Your ANU ID check code is:          ', md5_anu_id[:8])
print('Your student data set check code is:', md5_data_set[:12])
print()
print('  *** Check this pair of codes is in the list provided on Wattle')
print('  *** If not contact the course convenor.')
print()
print('#### ', md5_anu_id[:8],'/',md5_data_set[:12])
print()
