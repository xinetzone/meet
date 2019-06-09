import os
import tarfile

def untar(root):
    '''
    解压 .tar 文件
    '''
    for fname in os.listdir(root):
        t = tarfile.TarFile(os.path.join(root, fname))
        t.extractall(root)