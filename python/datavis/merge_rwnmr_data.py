import os
import shutil
import subprocess

DB_PATH = '/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/db'
MERGE_AUTH = 'merged_info.txt'

def create_merged_folder(foldername, folders_to_merge):
	complete_path = os.path.join(DB_PATH, foldername)
	print('Creating dir', complete_path)

	# Create dir
	success = False
	try:
		os.mkdir(complete_path)

		# Create file with merged folders info:
		merge_info_file = os.path.join(complete_path, MERGE_AUTH)
		with open(merge_info_file, 'w') as txt_file:
			txt_file.write('This folder was created by merging:\n')
			
			fcount = 0
			for folder in folders_to_merge:
				txt_file.write("f" + str(fcount) + ': "' + folder + '"\n')
				fcount += 1

		success = True
	
	except OSError as error:
		print(error)  
	
	return success

def parse_merge_authentication(destination):
	source_folders = []
	filepath = os.path.join(destination, MERGE_AUTH)

	with open(filepath, 'r') as file:
		next(file)
		lines = file.readlines()
		for line in lines:
			folder = line.split(': ')[-1][1:-2]
			source_folders.append(folder)
	
	return source_folders

def merge_folders(merged_folder, folders_to_merge):
	destination = os.path.join(DB_PATH, merged_folder)
	if(MERGE_AUTH in os.listdir(destination)):

		source_folders = parse_merge_authentication(destination)
		fcount = 0
		for folder in folders_to_merge:

			if(folder in source_folders):
				source = os.path.join(DB_PATH, folder)
				ftag = 'f' + str(fcount) + '_'
				copytree(source, destination, ftag)
				fcount += 1
	else:
		print("Destination is not a merging folder.")
	return

def copytree(src, dst, itemtag='', symlinks=False, ignore=None):
	if not os.path.exists(dst):
		os.makedirs(dst)

	for item in os.listdir(src):
		# print(item)
		# print(itemtag)
		ditem = itemtag + item
		s = os.path.join(src, item)
		d_dir = os.path.join(dst, ditem)
		d_file = os.path.join(dst, item)
		if os.path.isdir(s):
			copytree(s, d_dir, itemtag, symlinks, ignore)
		else:
			if not os.path.exists(d_file) or os.stat(s).st_mtime - os.stat(d_file).st_mtime > 1:
				shutil.copy2(s, d_file)

	return


# get list of rwnmr data folders from DB
folders = sorted(os.listdir(DB_PATH))
nFolders = len(folders)

# get merged folders from user input
print('>> Current folders in DB:')
for fIdx in range(nFolders):
	print(fIdx, '>', folders[fIdx])

user_inputs = input('>> Merge folders: ').split(' ')
valid_inputs = []
invalid_inputs = []
folders_to_merge = []
for user_input in user_inputs:
	try:
		fId = int(user_input)
		if not (fId < 0 or fId > nFolders):
			if not (fId in folders_to_merge):
				print(fId, '>', folders[fId])
				folders_to_merge.append(folders[fId])
	except:
		invalid_inputs.append(user_input)

	

if(len(folders_to_merge) > 0):
	new_folder_name = input('>> New folder name: ')
	
	print("\nMerging folder '", new_folder_name, "' will be created.")
	confirmation = input('>> Confirm? (Y/N): ')
	
	if(confirmation == 'Y' or confirmation == "y"):

		creation_is_confirmed = create_merged_folder(new_folder_name, folders_to_merge)
		
		if(creation_is_confirmed):
			
			print('\nFolders:')
			for folder in folders_to_merge:
				print(os.path.join(DB_PATH, folder))
			print('will be merged to:\n', os.path.join(DB_PATH, new_folder_name))
			
			confirmation = input('>> Confirm? (Y/N): ')
			
			if(confirmation == 'Y' or confirmation == "y"):
				print("Merging folders...")
				merge_folders(new_folder_name, folders_to_merge)
			else:
				print('Aborted.')
	else:
		print('Aborted.')
		
else:
	print('user input was invalid')