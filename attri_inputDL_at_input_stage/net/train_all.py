import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

os.system(f'python /home/wenzhang/AttributesandUnet_testsforPaper/test12_2/net/train.py')
os.system(f'python /home/wenzhang/AttributesandUnet_testsforPaper/test13/net/train.py')
os.system(f'python /home/wenzhang/AttributesandUnet_testsforPaper/test23/net/train.py')
os.system(f'python /home/wenzhang/AttributesandUnet_testsforPaper/test24/net/train.py')
os.system(f'python /home/wenzhang/AttributesandUnet_testsforPaper/test25/net/train.py')