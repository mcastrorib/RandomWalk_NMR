import os
import subprocess

def parse_config_files(CONFIG_DIR, dirpath):
	input_filelist = sorted(os.listdir(os.path.join(CONFIG_DIR, dirpath)))

	config_files = {}
	for file in input_filelist:
		if('uct.config' in file):
			config_files['-uctconfig'] = os.path.join(dirpath, file)
		if('pfgse.config' in file):
			config_files['-config'] = os.path.join(dirpath, file)
		if('rwnmr.config' in file):
			config_files['-rwconfig'] = os.path.join(dirpath, file)

	return config_files

def rename_file(old_name, new_name):
	# Set linux bash commands
	run_commands = ['mv', old_name, new_name]

	try:
		# Open subprocess to run RWNMR app
		process = subprocess.Popen(run_commands, 
								   stdout=subprocess.PIPE, 
								   stderr=subprocess.PIPE)
		
		# Check if process is completed
		stdout, stderr = process.communicate()
		
	except subprocess.CalledProcessError as e:
	    print('exit code: {}'.format(e.returncode))
	    print('stdout: {}'.format(e.output.decode(sys.getfilesystemencoding())))
	    print('stderr: {}'.format(e.stderr.decode(sys.getfilesystemencoding())))
	return

def run_rwnmr_pfgse(rwnmr_execpath, config_files, output_file, verbose=False):

	cmds = [rwnmr_execpath]
	paths = []
	for k in config_files:
		if(k == '-config'):
			cmds.append('pfgse')
		cmds.append(k)
		cmds.append(config_files[k])
	if('pfgse' not in cmds):
		cmds.append('pfgse')
	if(verbose):
		print(cmds)

	try:
		# Open subprocess to run chfem app
		process = subprocess.Popen(cmds, 
								   stdout=subprocess.PIPE, 
								   stderr=subprocess.PIPE)
		
		# Check if process is completed
		stdout, stderr = process.communicate()

		# Write results in output file
		lines = stdout.decode('utf-8').split('\n')
		with open(output_file, 'w') as file:
			write_line = False
			for line in lines:
				if(write_line == False and '>>> NMR SIMULATION 3D PARAMETERS:' in line):
					write_line = True
				if(write_line):
					file.write(line+'\n')

	except subprocess.CalledProcessError as e:
	    print('exit code: {}'.format(e.returncode))
	    print('stdout: {}'.format(e.output.decode(sys.getfilesystemencoding())))
	    print('stderr: {}'.format(e.stderr.decode(sys.getfilesystemencoding()))) 
	return

def main():
	# input parameters
	RWNMR_PATH = r'./RWNMR'
	DB_DIR = r'./db'
	CONFIG_DIR = r'./config'
	INPUTS_DIR = r'autorun'
	
	# Check if os paths are valid
	valid_paths = True
	if not (os.path.isfile(RWNMR_PATH)):
		valid_paths = False
	if not (os.path.isdir(DB_DIR)):
		valid_paths = False
	if not (os.path.isdir(CONFIG_DIR)):
		valid_paths = False
	if not (os.path.isdir(INPUTS_DIR)):
		valid_paths = False
	
	# Application main loop
	for new_input_dir in sorted(os.listdir(os.path.join(CONFIG_DIR, INPUTS_DIR))):
		print(':: Running', new_input_dir, 'analysis...')
		complete_path = os.path.join(INPUTS_DIR, new_input_dir)
		config_files = parse_config_files(CONFIG_DIR, complete_path)
		run_rwnmr_pfgse(RWNMR_PATH, config_files, os.path.join(CONFIG_DIR, complete_path, 'consolelog.txt'), verbose=False)	
		print('Completed.')

	return

if __name__ == '__main__':
	main()

