import matlab.engine
import sys
import os

sub_dir_name = sys.argv[1]
print(sub_dir_name)
eng = matlab.engine.start_matlab()
eng.eval('run initialize_matlab_variables.m', nargout=0)
matlab_input = {'input_path': os.getcwd(),
                'feat': ['h', 'rgb'],
                'output_path': os.path.join(os.getcwd(), 'output_path'),
                'sub_dir_name': sub_dir_name,
                'tissue_segment_dir': os.path.join(os.getcwd(), 'output_path')}
eng.workspace['matlab_input'] = matlab_input
eng.eval('run Pre_process_images.m', nargout=0)
