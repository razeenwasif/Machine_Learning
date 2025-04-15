import os

folder_path = '/Dataset'

for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        name, ext = os.path.splitext(filename)
        new_filename = f'digit{name}{ext}'

        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        os.rename(old_file, new_file)
print("Files renamed successfully")
